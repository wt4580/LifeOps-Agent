"""Harness 层：独立于 agent 业务逻辑的横切基础设施。

放在 ``common`` 而不是 ``agent`` 下，是为了让 ``common.config.llm_config``
可以依赖它而不触发 ``agent.__init__`` 的副作用（避免循环导入）。
"""
