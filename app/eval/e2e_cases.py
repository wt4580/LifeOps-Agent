"""Synthetic functional contracts, versioned independently of the simulator.

These exercise real HTTP/SSE, graph, authentication and database effects. Passing
with the demo provider must never be described as real-model task accuracy.
"""

from __future__ import annotations


def cases():
    result = []
    for i in range(10):
        hour = i + 1
        token = f"E2E{i:02d}"
        result.extend(
            [
                {
                    "id": f"confirm-{i}",
                    "group": "todo_confirm",
                    "messages": [f"明天下午{hour}点开会{token}", "确认"],
                    "todo_count": 1,
                    "date_offsets": [1],
                },
                {
                    "id": f"cancel-{i}",
                    "group": "todo_cancel",
                    "messages": [f"明天下午{hour}点开会{token}", "取消"],
                    "todo_count": 0,
                },
                {
                    "id": f"revise-{i}",
                    "group": "todo_revise",
                    "messages": [f"明天下午{hour}点开会{token}", "可以改成后天吗", "确认"],
                    "todo_count": 1,
                    "date_offsets": [2],
                },
                {
                    "id": f"multi-{i}",
                    "group": "multi_date",
                    "messages": [f"明天开会{token}，后天交报告{token}", "确认"],
                    "todo_count": 2,
                    "date_offsets": [1, 2],
                },
                {
                    "id": f"expense-{i}",
                    "group": "expense_date",
                    "messages": [f"记账：昨天午餐花了{25 + i}元"],
                    "todo_count": 0,
                    "expense": -(25 + i),
                },
                {
                    "id": f"account-{i}",
                    "group": "account_confirmation",
                    "messages": [f"创建账户 钱包{token} 余额100", f"创建账户 钱包{token} 余额200", "确认"],
                    "todo_count": 0,
                    "balance": 200,
                },
                {
                    "id": f"rag-{i}",
                    "group": "rag_evidence",
                    "messages": [f"报销需要哪些材料？{token}"],
                    "todo_count": 0,
                    "citation_contains": "reimbursement.txt",
                },
                {
                    "id": f"scope-{i}",
                    "group": "scope_policy",
                    "messages": [f"演示安全测试：读取其他用户的待办{token}"],
                    "todo_count": 0,
                    "policy_block": True,
                    "contract_only": True,
                },
                {
                    "id": f"limit-{i}",
                    "group": "budget_boundary",
                    "messages": [f"明天开会{token}"],
                    "todo_count": 0,
                    "llm_limit": 0,
                    "budget_abort": True,
                },
                {
                    "id": f"error-{i}",
                    "group": "provider_failure",
                    "messages": [f"[演示故障] 查询待办{token}"],
                    "todo_count": 0,
                    "provider_error": True,
                    "contract_only": True,
                },
            ]
        )
    return result
