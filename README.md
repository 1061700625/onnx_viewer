# ONNX 算子搜索工具

目录结构：

```text
onnx_op_search_split/
├── app.py
├── templates/
│   └── index.html
└── static/
    └── echarts.min.js   # 请放置本地 ECharts 文件
```

运行：

```bash
pip install flask onnx
python app.py
```

页面模板、CSS 和 JavaScript 位于 `templates/index.html`，ONNX 解析与 Flask API 位于 `app.py`。
