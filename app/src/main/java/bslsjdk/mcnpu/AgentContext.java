package bslsjdk.mcnpu;

import org.json.JSONArray;
import org.json.JSONObject;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class AgentContext {
    private AgentContext() {}

    /*
     * Long context is a routing problem as much as a token-budget problem.
     * Keep the complete history on-device, but build each prompt from four
     * complementary views:
     *
     *   origin  = the beginning of the conversation, USER first;
     *   spine   = important USER turns distributed across the full history;
     *   recall  = older turns relevant to the current USER request;
     *   recent  = the newest dialogue at full text.
     *
     * The user side is deliberately weighted above assistant chatter so the
     * model retains goals, constraints and decisions instead of prose noise.
     * No cloud call and no model-generated summary is required here.
     */
    public static String buildContext(JSONArray history, String current,
                                      int recentMessages, int maxChars) {
        if (history == null || history.length() == 0) return "";
        final int budget = Math.max(4096, maxChars);
        final StringBuilder out = new StringBuilder(Math.min(budget, 14000));
        final Set<Integer> used = new HashSet<>();

        appendSection(out, "Conversation continuity:",
                "Treat the USER history below as authoritative context. Preserve earlier goals, constraints, decisions and unfinished work unless the latest USER message explicitly changes them.");
        appendSection(out, "Conversation origin:",
                origin(history, used, 1800));
        appendSection(out, "User history spine:",
                userSpine(history, used, 2600));
        appendSection(out, "Relevant earlier conversation:",
                relevant(history, current, used, 2800));
        appendSection(out, "Recent conversation:",
                recent(history, Math.max(1, recentMessages), used, 3600));

        if (out.length() > budget) {
            return trim(out.toString(), budget);
        }
        return out.toString();
    }

    public static String buildRecent(JSONArray history, int maxMessages) {
        return recent(history, Math.max(1, maxMessages), new HashSet<>(), 12000);
    }

    private static String origin(JSONArray history, Set<Integer> used, int cap) {
        StringBuilder b = new StringBuilder();
        int found = 0;
        for (int i = 0; i < history.length() && found < 4; i++) {
            JSONObject m = history.optJSONObject(i);
            if (!isUser(m)) continue;
            appendMessage(b, m, 700, i, used);
            found++;
        }
        return trim(b.toString(), cap);
    }

    /*
     * Preserve USER turns across the whole conversation, not just the latest
     * window. First/last turns are anchors; evenly-spaced turns fill the middle.
     */
    private static String userSpine(JSONArray history, Set<Integer> used, int cap) {
        List<Integer> users = new ArrayList<>();
        for (int i = 0; i < history.length(); i++) {
            if (isUser(history.optJSONObject(i))) users.add(i);
        }
        if (users.isEmpty()) return "";

        LinkedHashSet<Integer> picks = new LinkedHashSet<>();
        int edgeCount = Math.min(3, users.size());
        for (int i = 0; i < edgeCount; i++) picks.add(users.get(i));
        for (int i = Math.max(edgeCount, users.size() - 3); i < users.size(); i++) {
            if (i >= 0) picks.add(users.get(i));
        }

        int target = Math.min(14, users.size());
        while (picks.size() < target) {
            double step = (double)(users.size() - 1) / Math.max(1, target - 1);
            int pos = (int)Math.round((picks.size()) * step);
            pos = Math.max(0, Math.min(users.size() - 1, pos));
            picks.add(users.get(pos));
            if (picks.size() == users.size()) break;
        }

        List<Integer> sorted = new ArrayList<>(picks);
        java.util.Collections.sort(sorted);

        StringBuilder b = new StringBuilder();
        for (int idx : sorted) {
            if (used.contains(idx)) continue;
            appendMessage(b, history.optJSONObject(idx), 320, idx, used);
        }
        return trim(b.toString(), cap);
    }

    private static String recent(JSONArray history, int count, Set<Integer> used, int cap) {
        StringBuilder b = new StringBuilder();
        int from = Math.max(0, history.length() - count);
        for (int i = from; i < history.length(); i++) {
            JSONObject m = history.optJSONObject(i);
            if (m == null || used.contains(i)) continue;
            appendMessage(b, m, 900, i, used);
        }
        return trim(b.toString(), cap);
    }

    /*
     * Score older USER messages much higher than ASSISTANT output. Relevance
     * also rewards exact multi-character/ASCII terms and recent position.
     */
    private static String relevant(JSONArray history, String current,
                                   Set<Integer> used, int cap) {
        if (current == null || current.trim().isEmpty()) return "";
        final Set<String> keys = keys(current);
        if (keys.isEmpty()) return "";

        List<Candidate> candidates = new ArrayList<>();
        int recentCut = Math.max(0, history.length() - 8);
        for (int i = 0; i < recentCut; i++) {
            JSONObject m = history.optJSONObject(i);
            if (m == null || used.contains(i)) continue;
            String text = m.optString("text", "");
            if (text.isEmpty()) continue;
            int score = relevance(text, keys);
            if (isUser(m)) score *= 4;
            else score *= 1;
            if (score > 0) {
                score += Math.min(6, i / 20);
                candidates.add(new Candidate(i, score));
            }
        }

        candidates.sort((a, b) -> {
            int byScore = Integer.compare(b.score, a.score);
            return byScore != 0 ? byScore : Integer.compare(a.index, b.index);
        });

        StringBuilder out = new StringBuilder();
        int selected = 0;
        for (Candidate c : candidates) {
            if (selected >= 8) break;
            if (used.contains(c.index)) continue;
            appendMessage(out, history.optJSONObject(c.index), isUser(history.optJSONObject(c.index)) ? 520 : 360,
                    c.index, used);
            selected++;
        }
        return trim(out.toString(), cap);
    }

    private static final class Candidate {
        final int index;
        final int score;
        Candidate(int index, int score) {
            this.index = index;
            this.score = score;
        }
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
            if (lower.contains(k)) score += k.length() >= 3 ? 3 : 1;
        }
        return score;
    }

    private static boolean isUser(JSONObject m) {
        return m != null && "user".equalsIgnoreCase(m.optString("role", ""));
    }

    private static boolean isCjk(char c) {
        return c >= 0x3400 && c <= 0x9fff;
    }

    private static void appendSection(StringBuilder out, String title, String body) {
        if (body == null || body.trim().isEmpty()) return;
        out.append(title).append('\n').append(body).append('\n');
    }

    private static void appendMessage(StringBuilder b, JSONObject m, int cap,
                                      int index, Set<Integer> used) {
        if (m == null || used.contains(index)) return;
        String role = m.optString("role", "assistant");
        String text = m.optString("text", "");
        if (text.isEmpty()) return;
        b.append('[').append(index).append("] ")
                .append(role).append(": ")
                .append(trim(text, cap))
                .append('\n');
        used.add(index);
    }

    private static String trim(String s, int cap) {
        if (s == null) return "";
        if (s.length() <= cap) return s;
        int head = Math.max(1, cap * 3 / 4);
        int tail = Math.max(1, cap - head);
        return s.substring(0, head) + " …[中间省略]… " + s.substring(s.length() - tail);
    }
}
