"""M9 网页搜索工具：360 抓取、正文提取、<tool_call> 解析。

分层约束：纯函数 + 网络，不碰 model_manager/session_store；
编排与落库在 chat/engine.py（技术方案 §1.1）。
"""
