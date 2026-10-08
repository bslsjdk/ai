package bslsjdk.mcnpu;

import org.json.JSONArray;
import org.json.JSONObject;

public final class AgentContext {
    private AgentContext() {}

    /*
     * The native executor has a bounded resident attention window, so blindly
     * sending the last N messages is not a real long-context strategy.
     * Keep three layers instead:
     *   1) origin: the beginning of the conversation, so the agent remembers why it started;
     *   2) ledger: coverage of the entire saved history, compacted without inventing facts;
     *   3) recent: the newest turns at full text.
     *
     * This is deliberately deterministic and local. The full history remains on-device;
     * this method only builds the small working set sent to the model.
     */
    public static String buildContext(JSONArray history, int recentMessages, int maxChars) {
        if (history == null || history.length() == 0) return "";
        final int budget = Math.max(4096, maxChars);
        final StringBuilder out = new StringBuilder(Math.min(budget, 12000));

        appendSection(out, "Conversation origin:", origin(history, Math.min(2, history.length()), 1800));
        appendSection(out, "Long-term conversation ledger:", ledger(history, 4200));
        appendSection(out, "Recent conversation:", recent(history, Math.max(1, recentMessages), 5200));

        if (out.length() > budget) {
            return out.substring(Math.max(0, out.length() - budget));
        }
        return out.toString();
    }

    // Compatibility for existing callers.
    public static String buildRecent(JSONArray history, int maxMessages) {
        return recent(history, Math.max(1, maxMessages), 12000);
    }

    private static String origin(JSONArray history, int count, int cap) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < count; i++) {
            appendMessage(b, history.optJSONObject(i), 900);
        }
        return trim(b.toString(), cap);
    }

    private static String recent(JSONArray history, int count, int cap) {
        StringBuilder b = new StringBuilder();
        int from = Math.max(0, history.length() - count);
        for (int i = from; i < history.length(); i++) {
            appendMessage(b, history.optJSONObject(i), 1200);
        }
        return trim(b.toString(), cap);
    }

    /*
     * Coverage sampling keeps the beginning/end and evenly spaced turns.
     * This prevents a 500-turn chat from becoming "I only remember the last 8".
     */
    private static String ledger(JSONArray history, int cap) {
        if (history.length() <= 12) return trim(recent(history, history.length(), cap), cap);

        StringBuilder b = new StringBuilder();
        int n = history.length();
        int slots = 12;
        for (int i = 0; i < slots; i++) {
            int idx = (int)Math.round((double)i * (n - 1) / (slots - 1));
            appendMessage(b, history.optJSONObject(idx), 360);
        }
        return trim(b.toString(), cap);
    }

    private static void appendSection(StringBuilder out, String title, String body) {
        if (body == null || body.trim().isEmpty()) return;
        out.append(title).append('\n').append(body).append('\n');
    }

    private static void appendMessage(StringBuilder b, JSONObject m, int cap) {
        if (m == null) return;
        String role = m.optString("role", "assistant");
        String text = m.optString("text", "");
        if (text.isEmpty()) return;
        b.append(role).append(": ").append(trim(text, cap)).append('\n');
    }

    private static String trim(String s, int cap) {
        if (s == null) return "";
        if (s.length() <= cap) return s;
        int head = Math.max(1, cap / 2);
        int tail = Math.max(1, cap - head);
        return s.substring(0, head) + " …[中间省略]… " + s.substring(s.length() - tail);
    }
}
