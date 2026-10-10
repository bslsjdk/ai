package bslsjdk.ornithnpu;

import org.junit.Test;
import static org.junit.Assert.*;

public class CharacterTokenizerTest {
    @Test public void fixedVocabularyRoundTripsKnownUnicodeCharacters() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("小狐狸🌱小树");
        int[] ids = tokenizer.encode("小狐狸🌱");
        assertEquals(CharacterTokenizer.VOCABULARY_SIZE, tokenizer.getVocabularySize());
        assertEquals("小狐狸🌱", tokenizer.decode(ids));
        for (int id : ids) assertTrue(id >= 0 && id < 200);
    }

    @Test public void unknownCharactersStayInsideVocabulary() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("森林");
        int[] ids = tokenizer.encode("森林海");
        assertEquals(0, ids[2]);
        assertEquals("森林□", tokenizer.decode(ids));
    }
}
