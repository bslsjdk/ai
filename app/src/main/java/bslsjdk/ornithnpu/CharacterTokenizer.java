package bslsjdk.ornithnpu;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.LinkedHashMap;
import java.util.Map;

/** Deterministic Unicode-code-point tokenizer for tiny character-level experiments. */
public final class CharacterTokenizer {
    public static final String FORMAT = "aimeng-char-tokenizer/v1";
    public static final String UNKNOWN = "\uFFFD";

    private final String[] tokens;
    private final Map<Integer, Integer> ids;

    private CharacterTokenizer(String[] tokens) {
        if (tokens == null || tokens.length < 2) {
            throw new IllegalArgumentException("tokenizer requires UNKNOWN plus at least one character");
        }
        this.tokens = tokens.clone();
        ids = new LinkedHashMap<>();
        for (int i = 1; i < this.tokens.length; i++) {
            int[] cps = this.tokens[i].codePoints().toArray();
            if (cps.length != 1 || ids.put(cps[0], i) != null) {
                throw new IllegalArgumentException("invalid or duplicate character token");
            }
        }
    }

    public static CharacterTokenizer fit(String text) {
        if (text == null || text.isEmpty()) throw new IllegalArgumentException("text cannot be empty");
        LinkedHashMap<Integer, Boolean> unique = new LinkedHashMap<>();
        text.codePoints().forEach(cp -> unique.put(cp, Boolean.TRUE));
        String[] tokens = new String[unique.size() + 1];
        tokens[0] = UNKNOWN;
        int index = 1;
        for (int cp : unique.keySet()) tokens[index++] = new String(Character.toChars(cp));
        return new CharacterTokenizer(tokens);
    }

    public int[] encode(String text) {
        if (text == null) throw new IllegalArgumentException("text cannot be null");
        int[] cps = text.codePoints().toArray();
        int[] result = new int[cps.length];
        for (int i = 0; i < cps.length; i++) result[i] = ids.getOrDefault(cps[i], 0);
        return result;
    }

    public String decode(int[] tokenIds) {
        if (tokenIds == null) throw new IllegalArgumentException("token IDs cannot be null");
        StringBuilder result = new StringBuilder();
        for (int id : tokenIds) {
            if (id < 0 || id >= tokens.length) throw new IllegalArgumentException("token ID out of range: " + id);
            if (id != 0) result.append(tokens[id]);
        }
        return result.toString();
    }

    public int getVocabularySize() { return tokens.length; }
    public String tokenAt(int id) {
        if (id < 0 || id >= tokens.length) throw new IllegalArgumentException("token ID out of range");
        return tokens[id];
    }

    public JSONObject toJson() throws JSONException {
        JSONObject root = new JSONObject();
        root.put("format", FORMAT);
        JSONArray array = new JSONArray();
        for (String token : tokens) array.put(token);
        root.put("tokens", array);
        return root;
    }

    public static CharacterTokenizer fromJson(JSONObject root) throws JSONException {
        if (root == null || !FORMAT.equals(root.optString("format", "")))
            throw new IllegalArgumentException("unsupported character tokenizer format");
        JSONArray array = root.getJSONArray("tokens");
        String[] tokens = new String[array.length()];
        for (int i = 0; i < tokens.length; i++) tokens[i] = array.getString(i);
        if (!UNKNOWN.equals(tokens[0])) throw new IllegalArgumentException("missing UNKNOWN token");
        return new CharacterTokenizer(tokens);
    }
}
