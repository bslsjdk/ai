package bslsjdk.ornithnpu;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Tiny Unicode-code-point tokenizer for on-device character prediction.
 * ID 0 is reserved for unknown characters; IDs 1..199 hold the first 199
 * distinct code points. The fixed output vocabulary is exactly 200 IDs.
 */
public final class CharacterTokenizer {
    public static final int VOCABULARY_SIZE = 200;
    private final Map<Integer, Integer> codePointToId = new LinkedHashMap<>();
    private final int[] idToCodePoint = new int[VOCABULARY_SIZE];

    public CharacterTokenizer(String seedText) {
        if (seedText == null) throw new IllegalArgumentException("seedText is null");
        int nextId = 1;
        for (int offset = 0; offset < seedText.length();) {
            int cp = seedText.codePointAt(offset);
            offset += Character.charCount(cp);
            if (!codePointToId.containsKey(cp) && nextId < VOCABULARY_SIZE) {
                codePointToId.put(cp, nextId);
                idToCodePoint[nextId] = cp;
                nextId++;
            }
        }
    }

    public int[] encode(String text) {
        if (text == null) throw new IllegalArgumentException("text is null");
        int count = text.codePointCount(0, text.length());
        int[] result = new int[count];
        int out = 0;
        for (int offset = 0; offset < text.length();) {
            int cp = text.codePointAt(offset);
            offset += Character.charCount(cp);
            Integer id = codePointToId.get(cp);
            result[out++] = id == null ? 0 : id;
        }
        return result;
    }

    public String decode(int[] ids) {
        if (ids == null) throw new IllegalArgumentException("ids is null");
        StringBuilder out = new StringBuilder(ids.length * 2);
        for (int id : ids) {
            if (id < 0 || id >= VOCABULARY_SIZE)
                throw new IllegalArgumentException("token ID out of vocabulary: " + id);
            if (id != 0 && idToCodePoint[id] != 0) out.appendCodePoint(idToCodePoint[id]);
            else out.append('□');
        }
        return out.toString();
    }

    public int getKnownCharacterCount() { return codePointToId.size(); }
    public int getVocabularySize() { return VOCABULARY_SIZE; }
}
