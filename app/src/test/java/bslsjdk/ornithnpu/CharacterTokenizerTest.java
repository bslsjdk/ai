package bslsjdk.ornithnpu;

import org.json.JSONObject;
import org.junit.Test;
import static org.junit.Assert.*;

public class CharacterTokenizerTest {
    @Test public void encodesAndDecodesUnicodeCharacters() throws Exception {
        String story = "小兔🐇爱分享。小鸟也开心。";
        CharacterTokenizer tokenizer = CharacterTokenizer.fit(story);
        int[] ids = tokenizer.encode(story);
        assertEquals(story.codePointCount(0, story.length()), ids.length);
        assertEquals(story, tokenizer.decode(ids));
        for (int id : ids) assertTrue(id >= 0 && id < tokenizer.getVocabularySize());
        CharacterTokenizer restored = CharacterTokenizer.fromJson(
                new JSONObject(tokenizer.toJson().toString()));
        assertArrayEquals(ids, restored.encode(story));
    }

    @Test public void unknownCharacterMapsToReservedUnknownId() {
        CharacterTokenizer tokenizer = CharacterTokenizer.fit("甲乙甲");
        assertEquals(0, tokenizer.encode("丙")[0]);
        assertEquals("甲乙甲", tokenizer.decode(tokenizer.encode("甲乙甲")));
    }
}
