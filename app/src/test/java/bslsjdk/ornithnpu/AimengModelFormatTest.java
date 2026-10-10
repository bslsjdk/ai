package bslsjdk.ornithnpu;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;

import static org.junit.Assert.*;

public class AimengModelFormatTest {
    @Rule public TemporaryFolder temp = new TemporaryFolder();

    @Test public void binaryBundleRoundTripsPortableJson() throws Exception {
        byte[] source = ("{\"format\":\"aimeng-mobile-diffusion-json-v1\","
                + "\"test\":\"你好 AIMENG\"}").getBytes(StandardCharsets.UTF_8);
        File json = new File(temp.getRoot(), "model.json");
        File aimg = new File(temp.getRoot(), "model.aimg");
        Files.write(json.toPath(), source);

        AimengModelFormat.packJsonFile(json, aimg);

        assertTrue(AimengModelFormat.isBinaryBundle(aimg));
        assertTrue(aimg.length() < source.length + 128);
        assertArrayEquals(source, AimengModelFormat.readJsonBytes(aimg));
        assertArrayEquals(source, AimengModelFormat.readJsonBytes(json));
    }

    @Test public void binaryBundleRejectsCorruptedPayload() throws Exception {
        byte[] source = "{\"format\":\"aimeng-mobile-diffusion-json-v1\"}".getBytes(StandardCharsets.UTF_8);
        File json = new File(temp.getRoot(), "model.json");
        File aimg = new File(temp.getRoot(), "model.aimg");
        Files.write(json.toPath(), source);
        AimengModelFormat.packJsonFile(json, aimg);
        byte[] bytes = Files.readAllBytes(aimg.toPath());
        bytes[bytes.length - 1] ^= 0x20;
        Files.write(aimg.toPath(), bytes);

        try {
            AimengModelFormat.readJsonBytes(aimg);
            fail("corrupted gzip payload must not be accepted");
        } catch (Exception expected) {
            assertNotNull(expected.getMessage());
        }
    }
}
