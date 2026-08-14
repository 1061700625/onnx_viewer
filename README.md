# ONNX 算子搜索工具

**目录结构：**

```text
onnx_op_search_split/
├── app.py
├── templates/
│   └── index.html
└── static/
    └── echarts.min.js
```

**运行：**

```bash
pip install flask onnx
python app.py
```

页面模板、CSS 和 JavaScript 位于 `templates/index.html`，ONNX 解析与 Flask API 位于 `app.py`。

**效果：**

<img width="3330" height="1710" alt="image" src="https://github.com/user-attachments/assets/c9bc3d5c-f919-47e4-bee3-9e2fb6cfe1a4" />

