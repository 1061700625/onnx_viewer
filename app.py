#!/usr/bin/env python3
import os
import onnx
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

current_model = None


def _decode_onnx_bytes(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)


def _tensor_attr_summary(tensor):
    try:
        dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
    except Exception:
        dtype = str(tensor.data_type)
    dims = list(tensor.dims)
    name = tensor.name or ''
    prefix = f"Tensor(name={name}, " if name else "Tensor("
    return f"{prefix}dtype={dtype}, shape={dims})"


def _graph_attr_summary(graph):
    return (
        f"Graph(name={graph.name or ''}, nodes={len(graph.node)}, "
        f"inputs={len(graph.input)}, outputs={len(graph.output)})"
    )


def _sparse_tensor_attr_summary(sparse_tensor):
    dims = list(sparse_tensor.dims)
    values = getattr(sparse_tensor, 'values', None)
    if values is not None:
        try:
            dtype = onnx.TensorProto.DataType.Name(values.data_type)
        except Exception:
            dtype = str(values.data_type)
    else:
        dtype = 'unknown'
    return f"SparseTensor(dtype={dtype}, shape={dims})"


def _type_proto_attr_summary(type_proto):
    text = str(type_proto).strip().replace('\n', ' ')
    return ' '.join(text.split())


def _attribute_value_text(attr):
    attr_type = attr.type
    if attr_type == onnx.AttributeProto.FLOAT:
        return repr(attr.f)
    if attr_type == onnx.AttributeProto.INT:
        return str(attr.i)
    if attr_type == onnx.AttributeProto.STRING:
        return _decode_onnx_bytes(attr.s)
    if attr_type == onnx.AttributeProto.TENSOR:
        return _tensor_attr_summary(attr.t)
    if attr_type == onnx.AttributeProto.GRAPH:
        return _graph_attr_summary(attr.g)
    if attr_type == onnx.AttributeProto.FLOATS:
        return '[' + ', '.join(repr(v) for v in attr.floats) + ']'
    if attr_type == onnx.AttributeProto.INTS:
        return '[' + ', '.join(str(v) for v in attr.ints) + ']'
    if attr_type == onnx.AttributeProto.STRINGS:
        return '[' + ', '.join(repr(_decode_onnx_bytes(v)) for v in attr.strings) + ']'
    if attr_type == onnx.AttributeProto.TENSORS:
        return '[' + ', '.join(_tensor_attr_summary(v) for v in attr.tensors) + ']'
    if attr_type == onnx.AttributeProto.GRAPHS:
        return '[' + ', '.join(_graph_attr_summary(v) for v in attr.graphs) + ']'
    if attr_type == onnx.AttributeProto.SPARSE_TENSOR:
        return _sparse_tensor_attr_summary(attr.sparse_tensor)
    if attr_type == onnx.AttributeProto.SPARSE_TENSORS:
        return '[' + ', '.join(_sparse_tensor_attr_summary(v) for v in attr.sparse_tensors) + ']'
    if attr_type == onnx.AttributeProto.TYPE_PROTO:
        return _type_proto_attr_summary(attr.tp)
    if attr_type == onnx.AttributeProto.TYPE_PROTOS:
        return '[' + ', '.join(_type_proto_attr_summary(v) for v in attr.type_protos) + ']'
    return '<未定义或未知属性类型>'


def _attribute_type_name(attr_type):
    try:
        return onnx.AttributeProto.AttributeType.Name(attr_type)
    except Exception:
        return str(attr_type)


def _model_opset_version(domain=''):
    """返回当前模型指定 domain 的 opset。ONNX 标准 domain 在 proto 中通常是空字符串。"""
    global current_model
    if current_model is None:
        return None
    normalized = '' if domain in ('', 'ai.onnx') else domain
    for opset in current_model.opset_import:
        if (opset.domain or '') == normalized:
            return int(opset.version)
    return None


def serialize_node_attributes(node):
    """
    返回两类属性：
    1. NodeProto 中真实序列化的显式/自定义 attribute。
    2. ONNX 标准 schema 中定义但模型未显式序列化的内置属性，包括默认值。

    onnx_tool 导出的 shape 模型经常不会把默认属性写入 NodeProto.attribute，
    因此仅遍历 node.attribute 会得到空列表或不完整结果。
    """
    result = []
    explicit_names = set()

    for attr in node.attribute:
        explicit_names.add(attr.name)
        item = {
            'name': attr.name,
            'type': _attribute_type_name(attr.type),
            'value': _attribute_value_text(attr),
            'source': '模型显式属性',
            'explicit': True,
        }
        if getattr(attr, 'ref_attr_name', ''):
            item['refAttrName'] = attr.ref_attr_name
        if getattr(attr, 'doc_string', ''):
            item['docString'] = attr.doc_string
        result.append(item)

    # 自定义 domain 可能没有注册到本机 ONNX schema，失败时保留显式属性即可。
    schema_domain = '' if (node.domain or '') in ('', 'ai.onnx') else node.domain
    opset = _model_opset_version(schema_domain)
    try:
        if opset is not None:
            schema = onnx.defs.get_schema(node.op_type, opset, schema_domain)
        else:
            schema = onnx.defs.get_schema(node.op_type, schema_domain)
    except Exception:
        schema = None

    if schema is not None:
        # 给显式属性补 schema 描述。
        schema_attrs = schema.attributes
        for item in result:
            schema_attr = schema_attrs.get(item['name']) if hasattr(schema_attrs, 'get') else None
            if schema_attr is not None and getattr(schema_attr, 'description', ''):
                item['schemaDescription'] = schema_attr.description

        for name, schema_attr in schema_attrs.items():
            if name in explicit_names:
                continue

            default_proto = getattr(schema_attr, 'default_value', None)
            has_default = (
                default_proto is not None and
                getattr(default_proto, 'type', onnx.AttributeProto.UNDEFINED) != onnx.AttributeProto.UNDEFINED
            )
            if has_default:
                value = _attribute_value_text(default_proto)
                source = 'ONNX Schema 默认值'
            else:
                value = '<模型未显式设置>'
                source = 'ONNX Schema 定义'

            item = {
                'name': name,
                'type': _attribute_type_name(getattr(schema_attr, 'type', onnx.AttributeProto.UNDEFINED)),
                'value': value,
                'source': source,
                'explicit': False,
                'required': bool(getattr(schema_attr, 'required', False)),
            }
            if getattr(schema_attr, 'description', ''):
                item['schemaDescription'] = schema_attr.description
            result.append(item)

    return result


def _tensor_type_summary_from_value_info(value_info):
    """把 ValueInfoProto 转成适合前端展示的 dtype/shape 摘要。"""
    info = {
        'name': value_info.name,
        'dtype': 'unknown',
        'shape': [],
        'shapeText': '?',
        'kind': 'value_info',
    }
    try:
        tp = value_info.type
        if tp.HasField('tensor_type'):
            tensor_type = tp.tensor_type
            if tensor_type.elem_type:
                try:
                    info['dtype'] = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
                except Exception:
                    info['dtype'] = str(tensor_type.elem_type)
            dims = []
            if tensor_type.HasField('shape'):
                for dim in tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        dims.append(int(dim.dim_value))
                    elif dim.HasField('dim_param') and dim.dim_param:
                        dims.append(dim.dim_param)
                    else:
                        dims.append('?')
            info['shape'] = dims
            info['shapeText'] = '[' + ', '.join(str(v) for v in dims) + ']'
        else:
            # Sequence/Map/Optional 等非 tensor 类型，保留 protobuf 的紧凑摘要。
            text = ' '.join(str(tp).strip().replace('\n', ' ').split())
            info['dtype'] = text or 'unknown'
            info['shapeText'] = '-'
    except Exception:
        pass
    return info


def build_tensor_info_map(graph):
    """
    汇总 graph.input/output/value_info/initializer。
    onnx_tool 的 shape 模型主要把推理结果持久化到 value_info。
    """
    result = {}

    def merge_value_info(value_info, kind):
        if not value_info.name:
            return
        info = _tensor_type_summary_from_value_info(value_info)
        info['kind'] = kind
        old = result.get(value_info.name)
        if old is None or old.get('shapeText') in ('?', '[]', '-'):
            result[value_info.name] = info
        else:
            # 保留更完整 shape，同时记录它属于 graph input/output/value_info。
            old['kind'] = kind if kind != 'value_info' else old.get('kind', kind)

    for value in graph.value_info:
        merge_value_info(value, 'value_info')
    for value in graph.input:
        merge_value_info(value, 'graph_input')
    for value in graph.output:
        merge_value_info(value, 'graph_output')

    for tensor in graph.initializer:
        if not tensor.name:
            continue
        try:
            dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
        except Exception:
            dtype = str(tensor.data_type)
        init_info = {
            'name': tensor.name,
            'dtype': dtype,
            'shape': list(tensor.dims),
            'shapeText': '[' + ', '.join(str(v) for v in tensor.dims) + ']',
            'kind': 'initializer',
        }
        # initializer 的 dtype/shape 通常最精确。
        result[tensor.name] = init_info

    return result


def get_model_export_info():
    global current_model
    if current_model is None:
        return {}
    return {
        'producerName': current_model.producer_name or '',
        'producerVersion': current_model.producer_version or '',
        'isOnnxTool': (current_model.producer_name or '').lower() == 'onnx_tool',
        'irVersion': int(current_model.ir_version),
        'opsets': [
            {
                'domain': opset.domain or 'ai.onnx',
                'version': int(opset.version),
            }
            for opset in current_model.opset_import
        ],
    }

def build_graph_index(graph):
    producer = {}
    consumers = {}
    for node in graph.node:
        for out in node.output:
            if out:  # ONNX 允许用空字符串表示未提供的可选输出
                producer[out] = node
        for inp in node.input:
            if inp:  # ONNX 允许用空字符串表示未提供的可选输入
                consumers.setdefault(inp, []).append(node)
    return producer, consumers


def get_subgraph_data(graph, target_node, depth):
    producer, consumers = build_graph_index(graph)
    graph_inputs = {value.name for value in graph.input if value.name}
    graph_outputs = {value.name for value in graph.output if value.name}
    initializers = {value.name for value in graph.initializer if value.name}
    tensor_info_map = build_tensor_info_map(graph)
    model_export_info = get_model_export_info()

    def tensor_info_for(tensor_name):
        info = tensor_info_map.get(tensor_name)
        if info is None:
            return {
                'name': tensor_name,
                'dtype': 'unknown',
                'shape': [],
                'shapeText': '?',
                'kind': 'unknown',
            }
        return dict(info)

    # 搜索目标节点的直接真实输入/输出必须完整展示。
    # 即使 depth=0，只要 input/output 对应实际 NodeProto，也至少展开 1 层，
    # 避免目标节点的多输入/多输出被误表示成虚拟节点。
    # 更深层的上下游仍按用户设置的 depth 控制。
    effective_depth = max(1, int(depth))

    # 先确定真实算子节点，不在遍历阶段压缩边。
    # 最后再按 tensor/input slot 重建边，避免同一对节点之间的多输入/多输出被合并。
    node_map = {}
    target_id = id(target_node)

    # 只有本次搜索命中的 target_node 享受“直接 I/O 全量实体化”的特殊规则。
    # 其他任何节点，即使 op_type 与目标节点相同，也仍按普通节点处理，
    # 搜索边界之外的输入/输出继续使用虚拟节点表示。
    def is_search_target(old_id):
        return old_id == target_id

    node_map[target_id] = {'node': target_node, 'level': 0}

    queue = [(target_node, 0)]
    while queue:
        curr, d = queue.pop(0)
        if d <= -effective_depth:
            continue
        for inp in curr.input:
            if not inp:
                continue
            p_node = producer.get(inp)
            if p_node is None:
                continue
            p_id = id(p_node)
            if p_id not in node_map:
                node_map[p_id] = {'node': p_node, 'level': d - 1}
                queue.append((p_node, d - 1))

    queue = [(target_node, 0)]
    while queue:
        curr, d = queue.pop(0)
        if d >= effective_depth:
            continue
        for out_name in curr.output:
            if not out_name:
                continue
            for c_node in consumers.get(out_name, []):
                c_id = id(c_node)
                if c_id not in node_map:
                    node_map[c_id] = {'node': c_node, 'level': d + 1}
                    queue.append((c_node, d + 1))

    id_mapping = {old_id: idx for idx, old_id in enumerate(node_map.keys())}

    def build_input_details(node):
        details = []
        for input_index, tensor_name in enumerate(node.input):
            if not tensor_name:
                continue
            p_node = producer.get(tensor_name)
            if p_node is not None:
                p_name = p_node.name or p_node.op_type
                if id(p_node) in node_map:
                    source_desc = f"子图内上游节点：{p_name}"
                else:
                    source_desc = f"子图外上游节点：{p_name}"
            elif tensor_name in initializers:
                source_desc = '初始化器'
            elif tensor_name in graph_inputs:
                source_desc = '图输入'
            else:
                source_desc = '来源未知'
            details.append({
                'index': input_index,
                'tensor': tensor_name,
                'source': source_desc,
                'tensorInfo': tensor_info_for(tensor_name),
            })
        return details

    def build_output_details(node):
        details = []
        for output_index, tensor_name in enumerate(node.output):
            if not tensor_name:
                continue
            all_consumers = consumers.get(tensor_name, [])
            visible = [c.name or c.op_type for c in all_consumers if id(c) in node_map]
            hidden = [c.name or c.op_type for c in all_consumers if id(c) not in node_map]
            parts = []
            if visible:
                parts.append('子图内下游：' + ', '.join(visible))
            if hidden:
                parts.append('子图外下游：' + ', '.join(hidden))
            if tensor_name in graph_outputs:
                parts.append('图输出')
            if not all_consumers and tensor_name not in graph_outputs:
                parts.append('无下游消费者')
            details.append({
                'index': output_index,
                'tensor': tensor_name,
                'targets': '；'.join(parts) if parts else '无下游信息',
                'tensorInfo': tensor_info_for(tensor_name),
            })
        return details

    final_nodes = []
    real_positions = {}
    level_counts = {}
    for old_id, info in node_map.items():
        node = info['node']
        lvl = info['level']
        count = level_counts.get(lvl, 0)
        level_counts[lvl] = count + 1
        x_pos = (count - 2) * 200
        y_pos = lvl * 150
        category = 0 if lvl == 0 else (1 if lvl < 0 else 2)
        node_id = id_mapping[old_id]
        real_positions[old_id] = (x_pos, y_pos)
        input_details = build_input_details(node)
        output_details = build_output_details(node)

        # “多输入/多输出”视觉标识只用于提示该方向存在虚拟节点。
        # 纯子图内部的多输入/多输出不额外标记。
        virtual_input_count = 0
        for tensor_name in node.input:
            if not tensor_name:
                continue
            p_node = producer.get(tensor_name)
            if p_node is None or id(p_node) not in node_map:
                virtual_input_count += 1

        virtual_output_count = 0
        total_output_branches = 0
        for tensor_name in node.output:
            if not tensor_name:
                continue
            all_consumers = consumers.get(tensor_name, [])
            hidden_consumer_count = sum(1 for c in all_consumers if id(c) not in node_map)
            virtual_output_count += hidden_consumer_count
            total_output_branches += len(all_consumers)
            if tensor_name in graph_outputs:
                virtual_output_count += 1
                total_output_branches += 1
            elif not all_consumers:
                virtual_output_count += 1
                total_output_branches += 1

        # 搜索目标节点的所有直接 I/O 都会在后面显式展开为真实算子节点或常驻边界节点，
        # 因此目标节点本身不使用“含虚拟输入/输出”的提示。
        if is_search_target(old_id):
            virtual_input_count = 0
            virtual_output_count = 0

        has_multi_input = len(input_details) > 1 and virtual_input_count > 0
        has_multi_output = total_output_branches > 1 and virtual_output_count > 0
        io_hints = []
        if has_multi_input:
            io_hints.append('多输入')
        if has_multi_output:
            io_hints.append('多输出')
        node_attributes = serialize_node_attributes(node)
        final_nodes.append({
            'id': node_id,
            'name': node.name or f"{node.op_type}_{old_id}",
            'type': node.op_type,
            'level': lvl,
            'category': category,
            'symbolSize': 45 if category == 0 else 30,
            'x': x_pos,
            'y': y_pos,
            'isVirtual': False,
            'isTarget': is_search_target(old_id),
            'inputCount': len(input_details),
            'outputCount': len(output_details),
            'outputBranchCount': total_output_branches,
            'hasMultiInput': has_multi_input,
            'hasMultiOutput': has_multi_output,
            'virtualInputCount': virtual_input_count,
            'virtualOutputCount': virtual_output_count,
            'ioHint': '、'.join(io_hints),
            'inputDetails': input_details,
            'outputDetails': output_details,
            'domain': node.domain or 'ai.onnx',
            'attributes': node_attributes,
            'attributeCount': len(node_attributes),
            'explicitAttributeCount': len(node.attribute),
            'modelExportInfo': model_export_info,
        })

    final_links = []

    # 真实边按“输入槽位”生成。即使 source/target 相同，也不会丢掉第二条连接。
    for dst_old_id, info in node_map.items():
        dst_node = info['node']
        dst_id = id_mapping[dst_old_id]
        for input_index, tensor_name in enumerate(dst_node.input):
            if not tensor_name:
                continue
            p_node = producer.get(tensor_name)
            if p_node is None:
                continue
            src_old_id = id(p_node)
            if src_old_id not in node_map:
                continue
            final_links.append({
                'source': id_mapping[src_old_id],
                'target': dst_id,
                'tensor': tensor_name,
                'inputIndex': input_index,
                'isVirtual': False,
            })

    # 仅搜索目标节点的直接 I/O 必须完整可见。
    # 对有真实 producer/consumer 的连接，上面已经生成真实算子节点和真实边。
    # 对 initializer / graph input / graph output / 悬空输出 / 无法解析来源，
    # 这里创建“常驻边界节点”。它们不是虚拟节点，不受“显示虚拟节点”开关控制。
    boundary_seq = 0
    target_x, target_y = real_positions[target_id]

    # 用可变容器保存序号，便于嵌套 helper 递增。
    boundary_seq_box = [0]

    def create_target_boundary_node(direction, slot_index, tensor_name, kind, detail,
                                    branch_index=0, branch_count=1):
        boundary_id = f"b{boundary_seq_box[0]}"
        boundary_seq_box[0] += 1

        # 搜索目标节点的常驻边界节点不占主线上下方向。
        # 输入源固定放在目标节点左侧，输出端固定放在右侧；
        # 同方向存在多个端口时沿纵向展开，避免与主链节点和彼此重叠。
        side_offset = 185
        spread = 68
        vertical_offset = (branch_index - (branch_count - 1) / 2.0) * spread
        if direction == 'input':
            x_pos = target_x - side_offset
            y_pos = target_y + vertical_offset
            lvl = -0.28
        else:
            x_pos = target_x + side_offset
            y_pos = target_y + vertical_offset
            lvl = 0.28
        category = 1 if direction == 'input' else 2
        short_label = f"in{slot_index}" if direction == 'input' else f"out{slot_index}"

        final_nodes.append({
            'id': boundary_id,
            'name': tensor_name,
            'type': kind,
            'level': lvl,
            'category': category,
            'symbolSize': 24,
            'symbol': 'circle',
            'x': x_pos,
            'y': y_pos,
            'isVirtual': False,
            'isTargetBoundary': True,
            'isBoundary': True,
            'boundaryKind': kind,
            'direction': direction,
            'slotIndex': slot_index,
            'tensor': tensor_name,
            'detail': detail,
            'shortLabel': short_label,
        })
        return boundary_id

    # 目标节点输入中，没有 producer NodeProto 的输入也要显式列出来，而不是虚拟化。
    target_missing_inputs = []
    for input_index, tensor_name in enumerate(target_node.input):
        if not tensor_name:
            continue
        p_node = producer.get(tensor_name)
        if p_node is not None:
            # 真实 producer 已由 effective_depth 至少展开一层。
            continue
        if tensor_name in initializers:
            kind = 'Initializer'
            detail = '初始化器'
        elif tensor_name in graph_inputs:
            kind = 'GraphInput'
            detail = '图输入'
        else:
            kind = 'ExternalInput'
            detail = '外部输入 / 无法解析来源'
        target_missing_inputs.append((input_index, tensor_name, kind, detail))

    for i, (input_index, tensor_name, kind, detail) in enumerate(target_missing_inputs):
        boundary_id = create_target_boundary_node(
            'input', input_index, tensor_name, kind, detail,
            branch_index=i, branch_count=len(target_missing_inputs)
        )
        final_links.append({
            'source': boundary_id,
            'target': id_mapping[target_id],
            'tensor': tensor_name,
            'inputIndex': input_index,
            'isVirtual': False,
            'isBoundary': True,
        })

    # 目标节点输出如果直接是 Graph Output 或没有 consumer，也显式列为常驻边界节点。
    target_boundary_outputs = []
    for output_index, tensor_name in enumerate(target_node.output):
        if not tensor_name:
            continue
        all_consumers = consumers.get(tensor_name, [])
        if tensor_name in graph_outputs:
            target_boundary_outputs.append((output_index, tensor_name, 'GraphOutput', '图输出'))
        elif not all_consumers:
            target_boundary_outputs.append((output_index, tensor_name, 'DanglingOutput', '无下游消费者'))

    for i, (output_index, tensor_name, kind, detail) in enumerate(target_boundary_outputs):
        boundary_id = create_target_boundary_node(
            'output', output_index, tensor_name, kind, detail,
            branch_index=i, branch_count=len(target_boundary_outputs)
        )
        final_links.append({
            'source': id_mapping[target_id],
            'target': boundary_id,
            'tensor': tensor_name,
            'outputIndex': output_index,
            'isVirtual': False,
            'isBoundary': True,
        })

    # 曲率统一在所有真实边、边界边和虚拟边都创建完成后分配。
    # 这样虚拟边也会参与避让，不会只处理早期生成的真实边。

    virtual_seq = 0

    def add_virtual_node(parent_old_id, direction, slot_index, tensor_name,
                         kind, detail='', branch_index=0, branch_count=1):
        nonlocal virtual_seq
        parent_x, parent_y = real_positions[parent_old_id]
        virtual_id = f"v{virtual_seq}"
        virtual_seq += 1

        # 虚拟节点也不占主线上下方向。
        # 输入虚拟节点放在对应真实节点左侧，输出虚拟节点放在右侧；
        # 同一真实节点存在多个虚拟端口时沿纵向展开。这样即使关闭引力布局，
        # 虚拟节点也不会压在主链节点或主链连线上。
        side_offset = 135
        spread = 46
        vertical_offset = (branch_index - (branch_count - 1) / 2.0) * spread
        if direction == 'input':
            x_pos = parent_x - side_offset
            y_pos = parent_y + vertical_offset
            lvl = node_map[parent_old_id]['level'] - 0.18
        else:
            x_pos = parent_x + side_offset
            y_pos = parent_y + vertical_offset
            lvl = node_map[parent_old_id]['level'] + 0.18
        category = 1 if direction == 'input' else 2
        short_label = f"in{slot_index}" if direction == 'input' else f"out{slot_index}"

        final_nodes.append({
            'id': virtual_id,
            'name': tensor_name,
            'type': 'Virtual',
            'level': lvl,
            'category': category,
            'symbolSize': 13,
            'symbol': 'diamond',
            'x': x_pos,
            'y': y_pos,
            'isVirtual': True,
            'virtualKind': kind,
            'direction': direction,
            'slotIndex': slot_index,
            'tensor': tensor_name,
            'detail': detail,
            'shortLabel': short_label,
        })
        return virtual_id

    # 对普通节点“没有真实算子节点可展示”的输入建立虚拟输入节点。
    # 对目标节点而言，真实 producer 已通过 effective_depth 至少展开 1 层，因此不会被虚拟化。
    # 这里主要保留 graph input、initializer、普通边界节点的深度外 producer、无法解析来源。
    for old_id, info in node_map.items():
        # 搜索目标节点的直接输入已经全部实体化，不再创建任何虚拟输入节点。
        if is_search_target(old_id):
            continue
        node = info['node']
        missing_inputs = []
        for input_index, tensor_name in enumerate(node.input):
            if not tensor_name:
                continue
            p_node = producer.get(tensor_name)
            if p_node is not None and id(p_node) in node_map:
                continue

            if tensor_name in initializers:
                kind = 'initializer'
                detail = '初始化器'
            elif tensor_name in graph_inputs:
                kind = 'graph_input'
                detail = '图输入'
            elif p_node is not None:
                kind = 'hidden_upstream'
                detail = p_node.name or p_node.op_type
            else:
                kind = 'external_input'
                detail = '外部输入 / 无法解析来源'
            missing_inputs.append((input_index, tensor_name, kind, detail))

        for i, (input_index, tensor_name, kind, detail) in enumerate(missing_inputs):
            virtual_id = add_virtual_node(
                old_id, 'input', input_index, tensor_name, kind, detail,
                branch_index=i, branch_count=len(missing_inputs)
            )
            final_links.append({
                'source': virtual_id,
                'target': id_mapping[old_id],
                'tensor': tensor_name,
                'inputIndex': input_index,
                'isVirtual': True,
                'lineStyle': {'type': 'dashed', 'width': 1.5, 'opacity': 0.6, 'curveness': 0},
            })

    # 对普通节点的搜索范围外 consumer、Graph Output 和悬空输出建立虚拟输出节点。
    # 目标节点的真实 consumer 已至少展开 1 层，不会被虚拟化。
    for old_id, info in node_map.items():
        # 搜索目标节点的直接输出已经全部实体化，不再创建任何虚拟输出节点。
        if is_search_target(old_id):
            continue
        node = info['node']
        missing_outputs = []
        for output_index, tensor_name in enumerate(node.output):
            if not tensor_name:
                continue

            all_consumers = consumers.get(tensor_name, [])
            hidden_consumers = [c for c in all_consumers if id(c) not in node_map]

            for c_node in hidden_consumers:
                missing_outputs.append((
                    output_index,
                    tensor_name,
                    'hidden_downstream',
                    c_node.name or c_node.op_type,
                ))

            if tensor_name in graph_outputs:
                missing_outputs.append((
                    output_index,
                    tensor_name,
                    'graph_output',
                    '图输出',
                ))
            elif not all_consumers:
                missing_outputs.append((
                    output_index,
                    tensor_name,
                    'dangling_output',
                    '无下游消费者',
                ))

        for i, (output_index, tensor_name, kind, detail) in enumerate(missing_outputs):
            virtual_id = add_virtual_node(
                old_id, 'output', output_index, tensor_name, kind, detail,
                branch_index=i, branch_count=len(missing_outputs)
            )
            final_links.append({
                'source': id_mapping[old_id],
                'target': virtual_id,
                'tensor': tensor_name,
                'outputIndex': output_index,
                'isVirtual': True,
                'lineStyle': {'type': 'dashed', 'width': 1.5, 'opacity': 0.6, 'curveness': 0},
            })

    # 所有边生成完成后统一分配曲率。
    # 目标是让残差/skip connection 这种跨层短接线明显从主链旁边岔开，而不是与主链重叠。
    # 处理顺序：跨层长边 -> 同端点并行边 -> 普通扇出 -> 普通扇入。
    def _edge_sort_key(link):
        return (
            str(link.get('target', '')),
            str(link.get('source', '')),
            int(link.get('inputIndex', -1)),
            int(link.get('outputIndex', -1)),
            str(link.get('tensor', '')),
        )

    node_visual_info = {node['id']: node for node in final_nodes}

    def _edge_level_span(link):
        src = node_visual_info.get(link.get('source'))
        tgt = node_visual_info.get(link.get('target'))
        if src is None or tgt is None:
            return 0.0
        try:
            return abs(float(tgt.get('level', 0)) - float(src.get('level', 0)))
        except (TypeError, ValueError):
            return 0.0

    # 真正跨过至少一个完整算子层的边视为 skip/residual 长边。
    # 虚拟节点通常位于 +/-0.48 层，不纳入这个判断。
    def _is_skip_edge(link):
        if link.get('isVirtual'):
            return False
        return _edge_level_span(link) > 1.05

    parallel_groups = {}
    outgoing_groups = {}
    incoming_groups = {}
    skip_edges = []
    for link in final_links:
        parallel_groups.setdefault((link['source'], link['target']), []).append(link)
        outgoing_groups.setdefault(link['source'], []).append(link)
        incoming_groups.setdefault(link['target'], []).append(link)
        if _is_skip_edge(link):
            skip_edges.append(link)

    # 先给每条边建立可修改的 lineStyle，并清掉旧曲率。
    for link in final_links:
        style = dict(link.get('lineStyle') or {})
        style['curveness'] = 0.0
        link['lineStyle'] = style

    # 1. 残差/skip connection 单独大幅岔开。
    #    单条短接默认走右侧；同一局部存在多条短接时左右交替，并逐渐增加弯曲幅度。
    #    这样主链边可以继续保持接近直线，不会与短接线贴在一起。
    skip_edge_ids = set()
    ordered_skips = sorted(
        skip_edges,
        key=lambda link: (
            -_edge_level_span(link),
            str(link.get('source', '')),
            str(link.get('target', '')),
            str(link.get('tensor', '')),
        )
    )
    for i, link in enumerate(ordered_skips):
        span = _edge_level_span(link)
        # 基础弯曲随跨层数增加。2 层约 0.30，跨得越远越往外绕。
        # 比上一版整体再向主链外侧拉开，给 residual/skip connection 留出更明显的视觉间隔。
        magnitude = min(0.60, 0.30 + max(0.0, span - 2.0) * 0.060)

        # 对与当前短接共享 source 或 target 的其他短接做左右交替，进一步避免互相覆盖。
        related_before = 0
        for prev in ordered_skips[:i]:
            if prev.get('source') == link.get('source') or prev.get('target') == link.get('target'):
                related_before += 1
        side = 1.0 if related_before % 2 == 0 else -1.0
        magnitude += min(0.18, (related_before // 2) * 0.06)
        link['lineStyle']['curveness'] = side * min(0.68, magnitude)
        # 残差线稍微加深一点，便于沿着曲线追踪，但不改变线型。
        link['lineStyle'].setdefault('opacity', 0.9)
        skip_edge_ids.add(id(link))

    # 2. 完全相同端点的并行边，使用对称曲率。
    #    已经属于 skip 的边不覆盖其大曲率。
    parallel_edge_ids = set()
    for group in parallel_groups.values():
        normal_group = [link for link in group if id(link) not in skip_edge_ids]
        if len(normal_group) <= 1:
            continue
        ordered = sorted(normal_group, key=_edge_sort_key)
        center = (len(ordered) - 1) / 2.0
        for i, link in enumerate(ordered):
            curve = (i - center) * 0.14
            if len(ordered) == 2:
                curve = -0.09 if i == 0 else 0.09
            link['lineStyle']['curveness'] = max(-0.36, min(0.36, curve))
            parallel_edge_ids.add(id(link))

    # 3. 普通同源扇出边做小幅分离。
    #    skip 边已经绕到主链外侧，不再参与这里的扇出计算，避免把主链一起拉弯。
    for group in outgoing_groups.values():
        candidates = [
            link for link in group
            if id(link) not in parallel_edge_ids
            and id(link) not in skip_edge_ids
        ]
        if len(candidates) <= 1:
            continue
        ordered = sorted(candidates, key=_edge_sort_key)
        center = (len(ordered) - 1) / 2.0
        for i, link in enumerate(ordered):
            curve = (i - center) * 0.065
            link['lineStyle']['curveness'] = max(-0.28, min(0.28, curve))

    # 4. 对仍为直线的普通同目标扇入边再做一次分散。
    for group in incoming_groups.values():
        candidates = [
            link for link in group
            if id(link) not in parallel_edge_ids
            and id(link) not in skip_edge_ids
            and abs(float(link['lineStyle'].get('curveness', 0.0))) < 1e-9
        ]
        if len(candidates) <= 1:
            continue
        ordered = sorted(candidates, key=_edge_sort_key)
        center = (len(ordered) - 1) / 2.0
        for i, link in enumerate(ordered):
            curve = -(i - center) * 0.065
            link['lineStyle']['curveness'] = max(-0.28, min(0.28, curve))

    return final_nodes, final_links


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/upload', methods=['POST'])
def upload_model():
    global current_model
    if 'model' not in request.files: return jsonify({'success': False, 'error': '未选择 ONNX 模型文件'})
    file = request.files['model']
    try:
        model_bytes = file.read()
        current_model = onnx.load_from_string(model_bytes) # 
        return jsonify({'success': True, 'filename': file.filename})
    except Exception as e: return jsonify({'success': False, 'error': str(e)})

@app.route('/search_candidates', methods=['POST'])
def search_candidates():
    global current_model
    if not current_model: return jsonify({'error': '请先上传 ONNX 模型'})
    data = request.json
    keyword = data.get('keyword', '').strip().lower()
    exact_match = bool(data.get('exact_match', False))
    graph = current_model.graph
    candidates = []
    for idx, n in enumerate(graph.node):
        node_name = (n.name or '').lower()
        op_type = (n.op_type or '').lower()
        if exact_match:
            name_match = bool(node_name) and keyword == node_name
            type_match = bool(op_type) and keyword == op_type
        else:
            name_match = bool(node_name) and keyword in node_name
            type_match = bool(op_type) and keyword in op_type
        if name_match or type_match:
            candidates.append({'index': idx, 'name': n.name or "未命名", 'type': n.op_type})
    if not candidates: return jsonify({'error': '未找到匹配节点'})
    return jsonify({'candidates': candidates})

@app.route('/visualize', methods=['POST'])
def visualize():
    global current_model
    if not current_model: return jsonify({'error': '请先上传 ONNX 模型'})
    data = request.json
    node_index, depth = data.get('node_index'), data.get('depth', 1)
    try:
        target_node = current_model.graph.node[node_index]
    except (IndexError, TypeError):
        return jsonify({'error': '节点索引无效'})
    nodes, links = get_subgraph_data(current_model.graph, target_node, depth)
    return jsonify({'nodes': nodes, 'links': links})

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    print("ONNX 算子搜索工具启动中，浏览器将自动打开 http://127.0.0.1:5000")
    Timer(1.5, open_browser).start()
    app.run(debug=True)
