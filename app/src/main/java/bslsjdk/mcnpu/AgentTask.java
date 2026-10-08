package bslsjdk.ornithnpu;

import org.json.JSONObject;
import java.util.UUID;

public final class AgentTask {
    public enum State { IDLE, THINKING, TOOL_CALL, WAITING_RESULT, OBSERVING, FINAL, FAILED }
    public final String id = UUID.randomUUID().toString();
    public final long createdAt = System.currentTimeMillis();
    public State state = State.THINKING;
    public String goal;
    public String workspace = "";
    public int steps = 0;

    public AgentTask(String goal) { this.goal = goal == null ? "" : goal; }

    public JSONObject toJson() {
        JSONObject o = new JSONObject();
        try {
            o.put("id", id); o.put("createdAt", createdAt);
            o.put("state", state.name()); o.put("goal", goal);
            o.put("workspace", workspace); o.put("steps", steps);
        } catch (Throwable ignored) {}
        return o;
    }
}
