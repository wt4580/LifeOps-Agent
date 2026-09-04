# -*- coding: utf-8 -*-
"""生活助手知识库评估集。

所有内容均来自 app 的自建知识库（E:\\work\\knowledge_for_test 下的 txt），
ground truth 由知识库作者标注，自洽、可复现。

数据结构（每条 case 是一个 dict）：
  - type: "retrieval" | "qa" | "route"
  - query: 用户输入的自然语言问题
  - expected: list[str]，期望命中的文件名（不含路径），用于检索评估
  - gold: list[str]，期望覆盖的关键点，用于 RAG 问答评估
  - expect_tool / expect_action: 任务路由评估用
"""

RETRIEVAL_CASES = [
    # ---- 健康领域 ----
    {
        "type": "retrieval",
        "query": "我身高170厘米体重76.4千克，帮我算一下BMI是多少，属于什么范围",
        "expected": ["health_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "一个身高170的成年人标准体重大概是多少",
        "expected": ["health_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "每天要吃多少热量才能安全地减脂，热量缺口应该控制在什么范围",
        "expected": ["health_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "男性腰围超过多少厘米就算腹部肥胖了",
        "expected": ["health_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "体检时空腹血糖超过多少需要警惕糖尿病的风险",
        "expected": ["health_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "我想增肌，力量训练应该遵循什么原则，推荐做哪些复合动作",
        "expected": ["health_fitness.txt"],
    },
    {
        "type": "retrieval",
        "query": "每周应该安排多少有氧运动才能达到建议的运动量",
        "expected": ["health_fitness.txt"],
    },
    {
        "type": "retrieval",
        "query": "跑步之后膝盖疼怎么办，怎么预防运动损伤",
        "expected": ["health_fitness.txt"],
    },
    # ---- 财务领域 ----
    {
        "type": "retrieval",
        "query": "每个月该怎么记账，把支出分成哪几类比较好",
        "expected": ["finance_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "50 30 20 预算法是什么意思，钱应该怎么分配",
        "expected": ["finance_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "我总忍不住乱买东西，有什么控制消费冲动的方法",
        "expected": ["finance_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "应急金应该准备多少钱，放哪里比较合适",
        "expected": ["finance_saving.txt"],
    },
    {
        "type": "retrieval",
        "query": "什么是定投，为什么要用指数基金做长期投资",
        "expected": ["finance_saving.txt"],
    },
    {
        "type": "retrieval",
        "query": "信用卡分期和网贷这种高息债务应该怎么处理",
        "expected": ["finance_saving.txt"],
    },
    # ---- 效率领域 ----
    {
        "type": "retrieval",
        "query": "怎么用四象限法给待办任务排优先级",
        "expected": ["productivity_todo.txt"],
    },
    {
        "type": "retrieval",
        "query": "GTD 方法的核心流程是什么，怎么用",
        "expected": ["productivity_todo.txt"],
    },
    {
        "type": "retrieval",
        "query": "每天最重要的三件事是什么方法，应该先做哪件",
        "expected": ["productivity_todo.txt"],
    },
    {
        "type": "retrieval",
        "query": "番茄工作法怎么用，一个番茄钟多长时间，怎么休息",
        "expected": ["productivity_focus.txt"],
    },
    {
        "type": "retrieval",
        "query": "什么是深度工作，怎么安排深度工作时段",
        "expected": ["productivity_focus.txt"],
    },
    {
        "type": "retrieval",
        "query": "经常刷短视频浪费时间，怎么减少这种低价值时间消耗",
        "expected": ["productivity_focus.txt"],
    },
    # ---- 学习领域 ----
    {
        "type": "retrieval",
        "query": "艾宾浩斯遗忘曲线说明什么，怎么安排复习间隔来对抗遗忘",
        "expected": ["learning_method.txt"],
    },
    {
        "type": "retrieval",
        "query": "主动回忆和费曼技巧分别是什么，为什么比死记硬背有效",
        "expected": ["learning_method.txt"],
    },
    {
        "type": "retrieval",
        "query": "每天刷几道 LeetCode 比较合适，一次刷多久",
        "expected": ["learning_leetcode.txt"],
    },
    {
        "type": "retrieval",
        "query": "两数之和这种配对问题一般用什么数据结构和思路",
        "expected": ["learning_leetcode.txt"],
    },
    {
        "type": "retrieval",
        "query": "链表题里快慢指针一般用来解决什么问题",
        "expected": ["learning_leetcode.txt"],
    },
    # ---- 饮食领域 ----
    {
        "type": "retrieval",
        "query": "人体需要的三大营养素是什么，各自主要来自哪些食物",
        "expected": ["cooking_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "减脂期每千克体重应该吃多少克蛋白质",
        "expected": ["cooking_base.txt"],
    },
    {
        "type": "retrieval",
        "query": "番茄炒蛋具体怎么做",
        "expected": ["cooking_recipes.txt"],
    },
    {
        "type": "retrieval",
        "query": "平时炒菜怎么少放油少放盐但是又有味道",
        "expected": ["cooking_recipes.txt"],
    },
    # ---- 购物领域 ----
    {
        "type": "retrieval",
        "query": "网购七天无理由退货是怎么规定的，有哪些商品不适用",
        "expected": ["shopping_warranty.txt"],
    },
    {
        "type": "retrieval",
        "query": "三包是什么，三包期内商品坏了可以怎么处理",
        "expected": ["shopping_warranty.txt"],
    },
    {
        "type": "retrieval",
        "query": "买电脑这类大件之前应该怎么判断自己是不是真的需要",
        "expected": ["shopping_decision.txt"],
    },
    {
        "type": "retrieval",
        "query": "买东西怎么比价和看评价，避免被虚假折扣骗了",
        "expected": ["shopping_decision.txt"],
    },
    # ---- 出行领域 ----
    {
        "type": "retrieval",
        "query": "安排旅行行程的时候怎么把景点按区域归类避免来回跑",
        "expected": ["travel_plan.txt"],
    },
    {
        "type": "retrieval",
        "query": "出门旅行收拾行李一定要带哪些证件和重要物品",
        "expected": ["travel_packing.txt"],
    },
    {
        "type": "retrieval",
        "query": "旅行途中贵重物品怎么保管更安全",
        "expected": ["travel_packing.txt"],
    },
    # ---- 职业领域 ----
    {
        "type": "retrieval",
        "query": "写简历的时候项目经历应该怎么描述，STAR 法则是什么",
        "expected": ["career_resume.txt"],
    },
    {
        "type": "retrieval",
        "query": "面试自我介绍应该讲多长时间，重点说什么",
        "expected": ["career_resume.txt"],
    },
    {
        "type": "retrieval",
        "query": "怎么给自己设定清晰的职业目标并拆成里程碑",
        "expected": ["career_plan.txt"],
    },
    # ---- 综合 / 跨文件协作（重点，测多文件检索与联想） ----
    {
        "type": "retrieval",
        "query": "我晚上总是睡不好，白天很没精神，晚上应该注意什么能睡得更好",
        "expected": ["life_habits.txt"],
    },
    {
        "type": "retrieval",
        "query": "我用 OKR 定了个目标，怎么把它拆成可以执行的待办进入每天的日程",
        "expected": ["personal_goals.txt"],
    },
    {
        "type": "retrieval",
        "query": "新习惯太难坚持，有什么降低门槛和把习惯绑在一起的方法",
        "expected": ["personal_goals.txt"],
    },
    {
        "type": "retrieval",
        "query": "三个月内我又想减重又想存钱还想提升工作效率，三个目标怎么一起管理不互相挤占",
        "expected": ["scenario_coach.txt", "finance_saving.txt", "productivity_todo.txt"],
    },
    {
        "type": "retrieval",
        "query": "正在准备求职面试，同时要刷算法题还要保持身体健康，时间怎么统筹安排",
        "expected": ["scenario_career.txt", "learning_leetcode.txt", "life_habits.txt"],
    },
    {
        "type": "retrieval",
        "query": "睡眠不足对学习和记忆有什么影响，为什么复习完要睡好觉",
        "expected": ["life_habits.txt", "learning_method.txt"],
    },
    {
        "type": "retrieval",
        "query": "每天怎么安排学习和锻炼的时间才更有效率，应该把难任务放在什么时候",
        "expected": ["productivity_focus.txt", "life_habits.txt"],
    },
]

QA_CASES = [
    {
        "type": "qa",
        "query": "我身高170厘米体重76.4千克，BMI是多少？属于哪个范围？",
        "expected": ["health_base.txt"],
        "gold": ["约 26.4", "超重"],
    },
    {
        "type": "qa",
        "query": "每天安全的减脂热量缺口是多少？",
        "expected": ["health_base.txt"],
        "gold": ["300 到 500 千卡", "不超过 1000 千卡"],
    },
    {
        "type": "qa",
        "query": "什么是 50 30 20 预算法？",
        "expected": ["finance_base.txt"],
        "gold": ["50% 必要开销", "30% 想要消费", "20% 储蓄还债"],
    },
    {
        "type": "qa",
        "query": "应急金一般要准备多少钱？",
        "expected": ["finance_saving.txt"],
        "gold": ["三到六个月日常开销"],
    },
    {
        "type": "qa",
        "query": "番茄工作法具体怎么安排？",
        "expected": ["productivity_focus.txt"],
        "gold": ["25 分钟专注", "5 分钟休息", "四个番茄钟后 15-30 分钟长休息"],
    },
    {
        "type": "qa",
        "query": "对抗遗忘应该怎么安排复习间隔？",
        "expected": ["learning_method.txt"],
        "gold": ["第 1/2/4/7/15 天复习", "间隔逐渐拉长"],
    },
]

ROUTE_CASES = [
    {
        "type": "route",
        "query": "帮我记一下今天中午午餐花了25块，记到餐饮分类",
        "expect_action": "record_expense",
        "gold": ["餐饮", "25"],
    },
    {
        "type": "route",
        "query": "我这个月一共花了多少钱，帮我统计一下",
        "expect_action": "query_expenses",
        "gold": ["统计"],
    },
    {
        "type": "route",
        "query": "我现在账户里一共有多少钱",
        "expect_action": "query_accounts",
        "gold": ["账户", "余额"],
    },
    {
        "type": "route",
        "query": "帮我建一个招商银行的账户",
        "expect_action": "create_account",
        "gold": ["账户"],
    },
    {
        "type": "route",
        "query": "我身高170体重76，算一下我的BMI，属不属于肥胖",
        "expect_action": "query_knowledge",
        "gold": ["BMI"],
    },
    {
        "type": "route",
        "query": "我们下周一到下周五每天晚上要学习两小时算法，帮我排进日程",
        "expect_action": "propose_todo",
        "gold": ["学习", "算法", "每天晚上"],
    },
    {
        "type": "route",
        "query": "查一下我这周还剩哪些待办没做",
        "expect_action": "query_todos",
        "gold": ["待办"],
    },
    {
        "type": "route",
        "query": "明天上海天气怎么样",
        "expect_action": "query_weather",
        "gold": ["上海", "天气"],
    },
    {
        "type": "route",
        "query": "这个周末我有什么安排",
        "expect_action": "query_calendar",
        "gold": ["日程", "周末"],
    },
    {
        "type": "route",
        "query": "早上好，今天想喝点什么呢",
        "expect_action": "normal_chat",
        "gold": ["闲聊"],
    },
]


ALL_CASES = RETRIEVAL_CASES + QA_CASES + ROUTE_CASES


def load_cases(case_type: str | None = None) -> list[dict]:
    cases = ALL_CASES
    if case_type:
        cases = [c for c in cases if c["type"] == case_type]
    return cases


def main() -> None:
    from collections import Counter

    c = Counter(x["type"] for x in ALL_CASES)
    print("total cases:", len(ALL_CASES), dict(c))


if __name__ == "__main__":
    main()
