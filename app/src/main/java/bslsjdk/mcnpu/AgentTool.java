package bslsjdk.mcnpu;

import org.json.JSONObject;

public interface AgentTool {
    String id();
    String description();
    AgentPermission permission();
    JSONObject execute(JSONObject args) throws Exception;
}
