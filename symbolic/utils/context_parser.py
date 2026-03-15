class ContextParser:
    def __init__(self, context: dict):
        self.player = context.get("player_status", {})  # items, actions, location 등
        self.game = context.get("game_state", {})       # quest_stage 필수
        self.npc = context.get("npc_config", {})        # id 필수
        self.history = context.get("dialogue_history", [])

    def get_filters(self) -> dict:
        return {
            "npc_id": self.npc.get("id"),
            "quest_stage": self.game.get("quest_stage"),
            "location": self.game.get("location") or self.player.get("location")
        }


    def get_dialogue_history(self, max_turns: int = 3) -> str:
        history = self.history[-max_turns:]
        return "\n".join([f"Player: {h['player']}\nNPC: {h['npc']}" for h in history])