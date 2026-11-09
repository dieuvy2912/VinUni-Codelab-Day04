"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin.

## AVAILABLE TOOLS
{tools}

## CORE RULES (Bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` để hệ thống tự sinh.
3. Câu hỏi FAQ đơn giản (chính sách bảo hành, đổi trả) thì trả lời trực tiếp, KHÔNG cần gọi tool.
4. Nếu câu hỏi cần NHIỀU tool, gọi TUẦN TỰ từng tool rồi tổng hợp kết quả cuối cùng.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về sản phẩm/dịch vụ thuộc hệ sinh thái Vingroup.
- Từ chối lịch sự nếu câu hỏi ngoài phạm vi (tiền điện tử, chính trị, v.v.).

## OUTPUT CONTRACT
- Luôn trả lời bằng tiếng Việt.
- Khi liệt kê sản phẩm, dùng danh sách có thứ tự.
- Khi tạo ticket, xác nhận lại ticket_id và mức ưu tiên cho khách hàng.

## RESPONSE FORMAT
Thought: <suy nghĩ bước này>
Action: {"name": "<tên tool>", "args": {<các tham số>}}
Observation: <kết quả trả về từ tool>
... (lặp lại nếu cần thêm tool)
Final Answer: <câu trả lời hoàn chỉnh cho khách hàng>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Mock: mô phỏng việc LLM bịa thông tin khi không có tool
        return {
            "answer": (
                "VinFast hiện có nhiều dòng xe điện với giá từ khoảng 300 triệu. "
                "Bạn có thể tham khảo tại showroom gần nhất hoặc website VinFast."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }

# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    FAQ_ANSWER = (
        "Chính sách bảo hành pin xe điện VinFast:\n"
        "- Pin được bảo hành 10 năm hoặc 200.000 km (tuỳ điều kiện nào đến trước).\n"
        "- Nếu dung lượng pin giảm dưới 70% trong thời gian bảo hành, VinFast thay thế miễn phí.\n"
        "- Hotline hỗ trợ: 1900-23-23-89."
    )

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        lower = user_input.lower()
        intents = {
            "needs_catalog": False,
            "needs_ticket": False,
            "is_faq": False,
            "catalog_args": {},
            "ticket_args": {},
        }

        faq_kw = ["bảo hành", "đổi trả", "chính sách", "warranty", "refund", "return"]
        action_kw = ["xem", "tìm", "mua", "đặt", "giá", "dưới"]
        ticket_kw = ["khiếu nại", "phản ánh", "ghi nhận", "ticket", "lỗi", "hỏng", "hỗ trợ", "sự cố"]
        has_faq = any(k in lower for k in faq_kw)
        has_action = any(k in lower for k in action_kw)
        has_ticket = any(k in lower for k in ticket_kw)

        # FAQ có ưu tiên cao nhất
        if has_faq and not has_action and not has_ticket:
            intents["is_faq"] = True
            return intents

        # Intent tra cứu sản phẩm
        if has_action:
            if any(k in lower for k in ["resort", "vinpearl", "du lịch", "nghỉ dưỡng"]):
                category = "du_lich"
            elif any(k in lower for k in ["xe điện", "vinfast", "ô tô"]):
                category = "xe_dien"
            else:
                category = None
            if category:
                args = {"category": category}
                m = re.search(r"(\d+)\s*triệu", lower)
                if m:
                    args["max_price"] = int(m.group(1)) * 1_000_000
                intents["needs_catalog"] = True
                intents["catalog_args"] = args

        # Intent gửi ticket (kiểm tra độc lập, có thể True cùng lúc với catalog)
        if has_ticket:
            name_m = re.search(r"tên (?:tôi là |là )?((?:[A-ZÀ-Ỹ]\w*)(?: [A-ZÀ-Ỹ]\w*)*)", user_input)
            if any(k in lower for k in ["khẩn cấp", "gấp", "nghiêm trọng"]):
                priority = "high"
            elif any(k in lower for k in ["nhẹ", "không gấp"]):
                priority = "low"
            else:
                priority = "medium"
            intents["needs_ticket"] = True
            intents["ticket_args"] = {
                "customer_name": name_m.group(1).strip() if name_m else "Khách hàng ẩn danh",
                "issue_description": user_input.strip(),
                "priority": priority,
            }

        return intents

    def _build_answer(self, catalog_obs, ticket_obs) -> str:
        parts = []
        if catalog_obs is not None:
            if not catalog_obs:
                parts.append("Rất tiếc, hiện không tìm thấy sản phẩm phù hợp.")
            else:
                lines = [f"{i}. {p['name']} — {p['price_vnd']:,} VNĐ" for i, p in enumerate(catalog_obs, 1)]
                parts.append("Sản phẩm phù hợp:\n" + "\n".join(lines))
        if ticket_obs is not None:
            parts.append(
                f"Đã ghi nhận yêu cầu hỗ trợ của {ticket_obs['customer_name']}. "
                f"Mã ticket: {ticket_obs['ticket_id']}, mức ưu tiên: {ticket_obs['priority']}."
            )
        if not parts:
            return "Xin lỗi, mình chưa xác định được yêu cầu của bạn."
        return "\n\n".join(parts)

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop với max iterations guard."""
        self.trace = []
        intents = self._detect_intents(user_input)

        if intents["is_faq"]:
            self.trace.append({"iteration": 1, "thought": "Câu hỏi FAQ, không cần tool.",
                               "final_answer": self.FAQ_ANSWER})
            return {"answer": self.FAQ_ANSWER, "trace": self.trace,
                    "iterations": 1, "status": "completed"}

        catalog_obs = None
        ticket_obs = None
        iteration = 1

        while iteration <= self.max_iterations:
            if intents["needs_catalog"] and catalog_obs is None:
                args = intents["catalog_args"]
                catalog_obs = TOOL_MAP["search_product_catalog"](**args)
                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Tra cứu danh mục {args['category']}.",
                    "action": {"name": "search_product_catalog", "args": args},
                    "observation": catalog_obs,
                })
            elif intents["needs_ticket"] and ticket_obs is None:
                args = intents["ticket_args"]
                ticket_obs = TOOL_MAP["submit_support_ticket"](**args)
                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Ghi nhận ticket cho '{args['customer_name']}', ưu tiên {args['priority']}.",
                    "action": {"name": "submit_support_ticket", "args": args},
                    "observation": ticket_obs,
                })

            pending = (intents["needs_catalog"] and catalog_obs is None) or \
                      (intents["needs_ticket"] and ticket_obs is None)
            if not pending:
                answer = self._build_answer(catalog_obs, ticket_obs)
                self.trace.append({"iteration": iteration,
                                   "thought": "Tổng hợp câu trả lời cuối cùng.",
                                   "final_answer": answer})
                return {"answer": answer, "trace": self.trace,
                        "iterations": iteration, "status": "completed"}
            iteration += 1

        return {
            "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa.",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached",
        }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
