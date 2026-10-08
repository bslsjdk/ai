package bslsjdk.ornithnpu;

import java.util.LinkedHashMap;
import java.util.Map;
import org.json.JSONObject;

public final class AgentToolRegistry {
    private final Map<String, AgentTool> tools = new LinkedHashMap<>();

    public synchronized void register(AgentTool tool) {
        if (tool == null || tool.id() == null || tool.id().isEmpty()) return;
        tools.put(tool.id(), tool);
    }

    public synchronized AgentTool get(String id) {
        return tools.get(id);
    }

    public synchronized JSONObject describe() {
        JSONObject out = new JSONObject();
        try {
            for (AgentTool tool : tools.values()) {
                JSONObject item = new JSONObject();
                item.put("description", tool.description());
                item.put("permission", tool.permission().name());
                out.put(tool.id(), item);
            }
        } catch (Throwable ignored) {}
        return out;
    }
}
