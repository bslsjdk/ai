package bslsjdk.mcnpu;

import org.json.JSONArray;
import org.json.JSONObject;

public final class AgentContext {
    private AgentContext() {}

    public static String buildRecent(JSONArray history, int maxMessages) {
        if (history == null || history.length() == 0) return "";
        int from = Math.max(0, history.length() - Math.max(1, maxMessages));
        StringBuilder b = new StringBuilder(2048);
        for (int i = from; i < history.length(); i++) {
            JSONObject m = history.optJSONObject(i);
            if (m == null) continue;
            String role = m.optString("role", "assistant");
            String text = m.optString("text", "");
            if (text.isEmpty()) continue;
            b.append(role).append(": ").append(text).append('\n');
        }
        return b.toString();
    }
}
