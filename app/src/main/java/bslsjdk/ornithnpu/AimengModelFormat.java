package bslsjdk.ornithnpu;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Canonical portable AIMENG mobile bundle.
 *
 * .aimg layout: magic[4]="AIMG", int32 version=1, int32 uncompressed length,
 * sha256[32] of UTF-8 JSON payload, then a GZIP-compressed JSON payload.
 * This is a portable inference package, not a PyTorch checkpoint. Legacy JSON
 * bundles remain readable while cloud exporters migrate to .aimg.
 */
public final class AimengModelFormat {
    private static final byte[] MAGIC = new byte[] {'A', 'I', 'M', 'G'};
    private static final int VERSION = 1;
    public static final long MAX_PACKED_BYTES = 32L * 1024L * 1024L;
    public static final int MAX_JSON_BYTES = 64 * 1024 * 1024;

    private AimengModelFormat() {}

    public static boolean isBinaryBundle(File file) throws IOException {
        if (file == null || !file.isFile() || file.length() < 4) return false;
        byte[] head = new byte[4];
        try (FileInputStream in = new FileInputStream(file)) {
            return in.read(head) == 4 && Arrays.equals(head, MAGIC);
        }
    }

    /** Returns UTF-8 JSON bytes for either canonical .aimg or legacy .json. */
    public static byte[] readJsonBytes(File file) throws Exception {
        if (file == null || !file.isFile() || file.length() <= 0
                || file.length() > MAX_PACKED_BYTES) {
            throw new IOException("AIMENG bundle missing or larger than 32 MiB");
        }
        byte[] packed = Files.readAllBytes(file.toPath());
        if (packed.length < 4 || !startsWithMagic(packed)) return packed;
        if (packed.length < 44) throw new IOException("truncated AIMG header");
        ByteBuffer header = ByteBuffer.wrap(packed).order(ByteOrder.BIG_ENDIAN);
        header.position(4);
        int version = header.getInt();
        int expectedLength = header.getInt();
        byte[] expectedHash = new byte[32];
        header.get(expectedHash);
        if (version != VERSION) throw new IOException("unsupported AIMG version: " + version);
        if (expectedLength <= 0 || expectedLength > MAX_JSON_BYTES) {
            throw new IOException("invalid uncompressed AIMG payload size");
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream(Math.min(expectedLength, 1024 * 1024));
        try (GZIPInputStream gzip = new GZIPInputStream(
                new ByteArrayInputStream(packed, 44, packed.length - 44))) {
            byte[] buffer = new byte[8192];
            int n;
            while ((n = gzip.read(buffer)) != -1) {
                if (out.size() + n > MAX_JSON_BYTES || out.size() + n > expectedLength) {
                    throw new IOException("AIMG payload exceeds declared size");
                }
                out.write(buffer, 0, n);
            }
        }
        byte[] json = out.toByteArray();
        if (json.length != expectedLength) throw new IOException("AIMG payload length mismatch");
        if (!MessageDigest.isEqual(expectedHash, sha256(json))) {
            throw new IOException("AIMG SHA-256 verification failed");
        }
        return json;
    }

    /** Packs an existing portable JSON bundle to .aimg using atomic replacement. */
    public static void packJsonFile(File jsonFile, File outputFile) throws Exception {
        if (jsonFile == null || !jsonFile.isFile() || jsonFile.length() <= 0
                || jsonFile.length() > MAX_JSON_BYTES) {
            throw new IOException("source JSON missing or larger than 64 MiB");
        }
        byte[] json = Files.readAllBytes(jsonFile.toPath());
        byte[] compressed;
        ByteArrayOutputStream payload = new ByteArrayOutputStream();
        try (GZIPOutputStream gzip = new GZIPOutputStream(payload)) {
            gzip.write(json);
        }
        compressed = payload.toByteArray();
        if (compressed.length + 44L > MAX_PACKED_BYTES) {
            throw new IOException("compressed AIMG bundle exceeds 32 MiB");
        }
        File parent = outputFile.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IOException("cannot create output directory");
        }
        File temp = new File(outputFile.getAbsolutePath() + ".tmp");
        try (FileOutputStream stream = new FileOutputStream(temp)) {
            ByteBuffer header = ByteBuffer.allocate(44).order(ByteOrder.BIG_ENDIAN);
            header.put(MAGIC).putInt(VERSION).putInt(json.length).put(sha256(json));
            stream.write(header.array());
            stream.write(compressed);
            stream.getFD().sync();
        }
        try {
            Files.move(temp.toPath(), outputFile.toPath(),
                    StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (java.nio.file.AtomicMoveNotSupportedException e) {
            Files.move(temp.toPath(), outputFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } finally {
            if (temp.exists()) temp.delete();
        }
    }

    private static boolean startsWithMagic(byte[] bytes) {
        return bytes.length >= 4 && bytes[0] == MAGIC[0] && bytes[1] == MAGIC[1]
                && bytes[2] == MAGIC[2] && bytes[3] == MAGIC[3];
    }

    private static byte[] sha256(byte[] bytes) throws Exception {
        return MessageDigest.getInstance("SHA-256").digest(bytes);
    }
}
