package bslsjdk.mcnpu;

import org.json.JSONArray;
import org.json.JSONObject;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class AgentContext {
    private AgentContext() {}

    /*
     * Long context is a memory-management problem, not an excuse to forget the
     * beginning of the conversation. The complete history stays on-device.
     * Each prompt contains:
     *   origin   = why the conversation started;
     *   relevant= old turns selected against the current user message;
     *   ledger   = deterministic coverage across the whole history;
     *   recent  = newest turns at full text.
     *
     * No cloud call and no model-generated summary are required here.
     */
    public static String buildContext(JSONArray history, String current,
                                      int recentMessages, int maxChars) {
        if (history == null || history.length() == 0) return "";
        final int budget = Math.max(4096, maxChars);
        final StringBuilder out = new StringBuilder(Math.min(budget, 12000));

        appendSection(out, "Conversation origin:",
                origin(history, Math.min(2, history.length()), 1600));
        appendSection(out, "Relevant earlier conversation:",
                relevant(history, current, 3000));
        appendSection(out, "Long-term conversation ledger:",
                ledger(history, 2600));
        appendSection(out, "Recent conversation:",
                recent(history, Math.max(1, recentMessages), 4200));

        if (out.length() > budget) {
            return out.substring(0, budget);
        }
        return out.toString();
    }

    public static String buildRecent(JSONArray history, int maxMessages) {
        return recent(history, Math.max(1, maxMessages), 12000);
    }

    private static String origin(JSONArray history, int count, int cap) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < count; i++) appendMessage(b, history.optJSONObject(i), 800);
        return trim(b.toString(), cap);
    }

    private static String recent(JSONArray history, int count, int cap) {
        StringBuilder b = new StringBuilder();
        int from = Math.max(0, history.length() - count);
        for (int i = from; i < history.length(); i++)
            appendMessage(b, history.optJSONObject(i), 1000);
        return trim(b.toString(), cap);
    }

    /*
     * Select older turns that share terms with the current request. ASCII words
     * use whitespace/punctuation boundaries; Chinese uses character bigrams.
     * This is intentionally small and deterministic for a phone.
     */
    private static String relevant(JSONArray history, String current, int cap) {
        if (current == null || current.trim().isEmpty()) return "";
        final Set<String> keys = keys(current);
        if (keys.isEmpty()) return "";

        List<Candidate> candidates = new ArrayList<>();
        int recentCut = Math.max(0, history.length() - 8);
        for (int i = 0; i < recentCut; i++) {
            JSONObject m = history.optJSONObject(i);
            if (m == null) continue;
            String text = m.optString("text", "");
            if (text.isEmpty()) continue;
            int score = relevance(text, keys);
            if (score > 0) candidates.add(new Candidate(i, score));
        }

        candidates.sort((a, b) -> Integer.compare(b.score, a.score));
        StringBuilder out = new StringBuilder();
        int used = 0;
        for (Candidate c : candidates) {
            if (used >= 6) break;
            appendMessage(out, history.optJSONObject(c.index), 500);
            used++;
        }
        return trim(out.toString(), cap);
    }

    private static final class Candidate {
        final int index, score;
        Candidate(int index, int score) { this.index = index; this.score = score; }
    }

    private static Set<String> keys(String s) {
        Set<String> out = new HashSet<>();
        StringBuilder ascii = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
                ascii.append(Character.toLowerCase(ch));
            } else {
                if (ascii.length() >= 2) out.add(ascii.toString());
                ascii.setLength(0);
                if (i + 1 < s.length() && isCjk(ch) && isCjk(s.charAt(i + 1)))
                    out.add(s.substring(i, i + 2));
            }
        }
        if (ascii.length() >= 2) out.add(ascii.toString());
        return out;
    }

    private static int relevance(String text, Set<String> keys) {
        String lower = text.toLowerCase();
        int score = 0;
        for (String k : keys) {
            if (lower.contains(k)) score += k.length() >= 3 ? 2 : 1;
        }
        return score;
    }

    private static boolean isCjk(char c) {
        return c >= 0x3400 && c <= 0x9fff;
    }

    private static String ledger(JSONArray history, int cap) {
        if (history.length() <= 10) return trim(recent(history, history.length(), cap), cap);
        StringBuilder b = new StringBuilder();
        int n = history.length();
        int slots = 10;
        for (int i = 0; i < slots; i++) {
            int idx = (int)Math.round((double)i * (n - 1) / (slots - 1));
            appendMessage(b, history.optJSONObject(idx), 300);
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
