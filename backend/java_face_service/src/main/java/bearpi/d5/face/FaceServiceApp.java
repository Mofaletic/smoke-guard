package bearpi.d5.face;

import cn.smartjavaai.common.cv.SmartImageFactory;
import cn.smartjavaai.common.entity.DetectionInfo;
import cn.smartjavaai.common.entity.DetectionResponse;
import cn.smartjavaai.common.entity.R;
import cn.smartjavaai.common.entity.face.FaceInfo;
import cn.smartjavaai.common.enums.DeviceEnum;
import cn.smartjavaai.face.config.FaceDetConfig;
import cn.smartjavaai.face.config.FaceRecConfig;
import cn.smartjavaai.face.constant.FaceDetectConstant;
import cn.smartjavaai.face.enums.FaceDetModelEnum;
import cn.smartjavaai.face.enums.FaceRecModelEnum;
import cn.smartjavaai.face.factory.FaceDetModelFactory;
import cn.smartjavaai.face.factory.FaceRecModelFactory;
import cn.smartjavaai.face.model.facedect.FaceDetModel;
import cn.smartjavaai.face.model.facerec.FaceRecModel;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.stream.Stream;

public class FaceServiceApp {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final float MATCH_THRESHOLD = 0.72f;
    private static final int PORT = Integer.parseInt(System.getenv().getOrDefault("FACE_SERVICE_PORT", "8787"));
    private static final Path DATA_DIR = Paths.get(System.getenv().getOrDefault("FACE_SERVICE_DATA_DIR", "data"));
    private static final Path REGISTRY_FILE = DATA_DIR.resolve("face-registry.json");
    private static final String MODEL_MODE = System.getenv().getOrDefault("FACE_SERVICE_MODEL_MODE", "auto");
    private static final String ONLINE_MODEL_DIR = System.getenv().getOrDefault("FACE_SERVICE_MODEL_DIR", "").trim();
    private static final String SEETA_MODEL_DIR = System.getenv().getOrDefault("FACE_SERVICE_SEETA_MODEL_DIR", "").trim();

    private static final Map<String, StoredProfile> PROFILES = new ConcurrentHashMap<String, StoredProfile>();
    private static FaceRecModel faceRecModel;

    public static void main(String[] args) throws Exception {
        Files.createDirectories(DATA_DIR);
        System.setProperty("DJL_CACHE_DIR", DATA_DIR.resolve("djl-cache").toAbsolutePath().toString());
        SmartImageFactory.setEngine(SmartImageFactory.Engine.OPENCV);
        loadRegistry();
        faceRecModel = createFaceRecModel();

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", PORT), 0);
        server.createContext("/health", exchange -> writeJson(exchange, 200, mapOf(
                "ok", true,
                "profiles", PROFILES.size(),
                "threshold", MATCH_THRESHOLD
        )));
        server.createContext("/profiles", new ProfilesHandler());
        server.createContext("/enroll", new EnrollHandler());
        server.createContext("/register", new EnrollHandler());
        server.createContext("/match", new MatchHandler());
        server.createContext("/login", new MatchHandler());
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("Java face service listening on http://127.0.0.1:" + PORT);
    }

    private static FaceRecModel createFaceRecModel() {
        if ("offline".equalsIgnoreCase(MODEL_MODE)) {
            return createOfflineFaceRecModel();
        }

        RuntimeException primaryError = null;
        try {
            return createOnlineFaceRecModel();
        } catch (RuntimeException error) {
            primaryError = error;
            System.err.println("Primary DJL face model load failed, fallback to SeetaFace6 offline model.");
            error.printStackTrace();
        }

        try {
            return createOfflineFaceRecModel();
        } catch (RuntimeException fallbackError) {
            if (primaryError != null) {
                fallbackError.addSuppressed(primaryError);
            }
            throw fallbackError;
        }
    }

    private static FaceRecModel createOnlineFaceRecModel() {
        Path detModelPath = resolveOnlineModelPath("ultranet.pt");
        Path recModelPath = resolveOnlineModelPath("face_feature.pt");
        FaceDetConfig detConfig = new FaceDetConfig();
        detConfig.setModelEnum(FaceDetModelEnum.ULTRA_LIGHT_FAST_GENERIC_FACE);
        detConfig.setConfidenceThreshold(0.45f);
        detConfig.setNmsThresh(FaceDetectConstant.NMS_THRESHOLD);
        detConfig.setDevice(DeviceEnum.CPU);
        if (detModelPath != null) {
            detConfig.setModelPath(detModelPath.toString());
            System.out.println("Using cached local face detection model: " + detModelPath);
        }
        FaceDetModel detectModel = FaceDetModelFactory.getInstance().getModel(detConfig);

        FaceRecConfig recConfig = new FaceRecConfig();
        recConfig.setModelEnum(FaceRecModelEnum.FACENET_MODEL);
        recConfig.setCropFace(true);
        recConfig.setAlign(true);
        recConfig.setDevice(DeviceEnum.CPU);
        recConfig.setDetectModel(detectModel);
        if (recModelPath != null) {
            recConfig.setModelPath(recModelPath.toString());
            System.out.println("Using cached local face recognition model: " + recModelPath);
        }
        return FaceRecModelFactory.getInstance().getModel(recConfig);
    }

    private static FaceRecModel createOfflineFaceRecModel() {
        Path seetaModelPath = resolveSeetaModelPath();
        if (seetaModelPath == null) {
            throw new IllegalStateException(
                    "SeetaFace6 offline model directory was not found. Set FACE_SERVICE_SEETA_MODEL_DIR " +
                            "to a folder containing face_detector.csta, face_landmarker_pts5.csta and face_recognizer.csta."
            );
        }
        FaceDetConfig detConfig = new FaceDetConfig();
        detConfig.setModelEnum(FaceDetModelEnum.SEETA_FACE6_MODEL);
        detConfig.setConfidenceThreshold(0.45f);
        detConfig.setNmsThresh(FaceDetectConstant.NMS_THRESHOLD);
        detConfig.setDevice(DeviceEnum.CPU);
        detConfig.setModelPath(seetaModelPath.toString());
        FaceDetModel detectModel = FaceDetModelFactory.getInstance().getModel(detConfig);

        FaceRecConfig recConfig = new FaceRecConfig();
        recConfig.setModelEnum(FaceRecModelEnum.SEETA_FACE6_MODEL);
        recConfig.setCropFace(true);
        recConfig.setAlign(true);
        recConfig.setDevice(DeviceEnum.CPU);
        recConfig.setDetectModel(detectModel);
        recConfig.setModelPath(seetaModelPath.toString());
        return FaceRecModelFactory.getInstance().getModel(recConfig);
    }

    private static Path resolveOnlineModelPath(String fileName) {
        List<Path> candidates = new ArrayList<Path>();
        addCandidate(candidates, ONLINE_MODEL_DIR);
        addCandidate(candidates, System.getProperty("DJL_CACHE_DIR"));
        addCandidate(candidates, System.getenv("DJL_CACHE_DIR"));
        addCandidate(candidates, Paths.get(System.getProperty("user.home"), "smartjavaai_cache").toString());
        addCandidate(candidates, DATA_DIR.resolve("djl-cache").toString());

        for (Path candidate : candidates) {
            Path resolved = findModelDirectory(candidate, fileName);
            if (resolved != null) {
                return resolved;
            }
        }
        return null;
    }

    private static Path resolveSeetaModelPath() {
        List<Path> candidates = new ArrayList<Path>();
        addCandidate(candidates, SEETA_MODEL_DIR);
        addCandidate(candidates, ONLINE_MODEL_DIR);
        addCandidate(candidates, DATA_DIR.resolve("seeta").toString());
        addCandidate(candidates, DATA_DIR.resolve("models").toString());
        addCandidate(candidates, Paths.get(System.getProperty("user.home"), "smartjavaai_cache", "seeta").toString());

        for (Path candidate : candidates) {
            if (candidate == null || !Files.isDirectory(candidate)) {
                continue;
            }
            if (hasFile(candidate, "face_detector.csta")
                    && hasFile(candidate, "face_landmarker_pts5.csta")
                    && hasFile(candidate, "face_recognizer.csta")) {
                return candidate;
            }
        }
        return null;
    }

    private static void addCandidate(List<Path> candidates, String value) {
        if (value == null || value.trim().isEmpty()) {
            return;
        }
        Path candidate = Paths.get(value.trim()).toAbsolutePath().normalize();
        if (!candidates.contains(candidate)) {
            candidates.add(candidate);
        }
    }

    private static boolean hasFile(Path directory, String fileName) {
        return Files.isRegularFile(directory.resolve(fileName));
    }

    private static Path findModelDirectory(Path root, String fileName) {
        if (root == null || !Files.exists(root)) {
            return null;
        }
        if (Files.isRegularFile(root.resolve(fileName))) {
            return root;
        }
        try (Stream<Path> stream = Files.walk(root, 12)) {
            Path file = stream
                    .filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().equalsIgnoreCase(fileName))
                    .findFirst()
                    .orElse(null);
            return file == null ? null : file.getParent();
        } catch (IOException ignored) {
            return null;
        }
    }

    private static class ProfilesHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            addCors(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                writeEmpty(exchange, 204);
                return;
            }
            if ("GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                List<Map<String, Object>> profiles = new ArrayList<Map<String, Object>>();
                for (StoredProfile profile : PROFILES.values()) {
                    profiles.add(profile.toResponseMap());
                }
                writeJson(exchange, 200, mapOf("profiles", profiles));
                return;
            }
            if ("DELETE".equalsIgnoreCase(exchange.getRequestMethod())) {
                PROFILES.clear();
                saveRegistry();
                writeJson(exchange, 200, mapOf("ok", true));
                return;
            }
            writeJson(exchange, 405, mapOf("error", "method_not_allowed"));
        }
    }

    private static class EnrollHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            addCors(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                writeEmpty(exchange, 204);
                return;
            }
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                writeJson(exchange, 405, mapOf("error", "method_not_allowed"));
                return;
            }

            JsonObject body = readJson(exchange.getRequestBody());
            String imageBase64 = firstNonEmpty(
                    stringValue(body, "imageBase64"),
                    stringValue(body, "image")
            );
            String userId = firstNonEmpty(
                    stringValue(body, "userId"),
                    stringValue(body, "name")
            );
            String name = stringValue(body, "name");
            String role = stringValue(body, "role");
            String pin = stringValue(body, "pin");

            if (imageBase64.isEmpty()) {
                writeJson(exchange, 400, mapOf("error", "missing_image"));
                return;
            }
            if (userId.isEmpty()) {
                writeJson(exchange, 400, mapOf("error", "missing_user_id"));
                return;
            }

            float[] feature = extractFeature(imageBase64);
            if (feature == null) {
                writeJson(exchange, 422, mapOf("error", "face_not_detected"));
                return;
            }

            StoredProfile profile = new StoredProfile();
            profile.id = userId;
            profile.userId = userId;
            profile.name = name.isEmpty() ? userId : name;
            profile.role = role.isEmpty() ? "resident" : role;
            profile.pin = pin;
            profile.feature = feature;

            PROFILES.put(profile.id, profile);
            saveRegistry();
            writeJson(exchange, 200, mapOf(
                    "ok", true,
                    "success", true,
                    "userId", profile.userId,
                    "profile", profile.toResponseMap()
            ));
        }
    }

    private static class MatchHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            addCors(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                writeEmpty(exchange, 204);
                return;
            }
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                writeJson(exchange, 405, mapOf("error", "method_not_allowed"));
                return;
            }
            if (PROFILES.isEmpty()) {
                writeJson(exchange, 200, mapOf(
                        "matched", false,
                        "success", false,
                        "reason", "no_profiles"
                ));
                return;
            }

            JsonObject body = readJson(exchange.getRequestBody());
            String imageBase64 = firstNonEmpty(
                    stringValue(body, "imageBase64"),
                    stringValue(body, "image")
            );
            if (imageBase64.isEmpty()) {
                writeJson(exchange, 400, mapOf("error", "missing_image"));
                return;
            }

            float[] probe = extractFeature(imageBase64);
            if (probe == null) {
                writeJson(exchange, 200, mapOf(
                        "matched", false,
                        "success", false,
                        "reason", "face_not_detected"
                ));
                return;
            }

            StoredProfile bestProfile = null;
            float bestScore = -1f;
            for (StoredProfile profile : PROFILES.values()) {
                float score = cosineSimilarity(probe, profile.feature);
                if (score > bestScore) {
                    bestScore = score;
                    bestProfile = profile;
                }
            }

            boolean matched = bestProfile != null && bestScore >= MATCH_THRESHOLD;
            writeJson(exchange, 200, mapOf(
                    "matched", matched,
                    "success", matched,
                    "score", bestScore,
                    "confidence", bestScore,
                    "threshold", MATCH_THRESHOLD,
                    "userId", matched && bestProfile != null ? bestProfile.userId : null,
                    "profile", matched ? bestProfile.toResponseMap() : null
            ));
        }
    }

    private static float[] extractFeature(String dataUrl) {
        try {
            byte[] imageBytes = decodeDataUrl(dataUrl);
            R<DetectionResponse> response = faceRecModel.extractFeatures(imageBytes);
            if (response == null || !response.isSuccess() || response.getData() == null || response.getData().getDetectionInfoList() == null) {
                return null;
            }

            DetectionInfo best = response.getData().getDetectionInfoList()
                    .stream()
                    .filter(info -> info != null && info.getFaceInfo() != null && info.getFaceInfo().getFeature() != null)
                    .max(Comparator.comparing(DetectionInfo::getScore))
                    .orElse(null);
            if (best == null) {
                return null;
            }
            FaceInfo faceInfo = best.getFaceInfo();
            return faceInfo.getFeature();
        } catch (Exception error) {
            error.printStackTrace();
            return null;
        }
    }

    private static float cosineSimilarity(float[] left, float[] right) {
        if (left == null || right == null || left.length != right.length || left.length == 0) {
            return -1f;
        }
        double dot = 0;
        double leftNorm = 0;
        double rightNorm = 0;
        for (int index = 0; index < left.length; index += 1) {
            dot += left[index] * right[index];
            leftNorm += left[index] * left[index];
            rightNorm += right[index] * right[index];
        }
        if (leftNorm == 0 || rightNorm == 0) {
            return -1f;
        }
        return (float) (dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm)));
    }

    private static byte[] decodeDataUrl(String dataUrl) {
        String payload = dataUrl;
        int comma = payload.indexOf(',');
        if (comma >= 0) {
            payload = payload.substring(comma + 1);
        }
        return Base64.getDecoder().decode(payload);
    }

    private static void loadRegistry() throws IOException {
        if (!Files.exists(REGISTRY_FILE)) {
            return;
        }
        String json = new String(Files.readAllBytes(REGISTRY_FILE), StandardCharsets.UTF_8);
        RegistryFile registry = GSON.fromJson(json, RegistryFile.class);
        if (registry == null || registry.profiles == null) {
            return;
        }
        PROFILES.clear();
        for (StoredProfile profile : registry.profiles) {
            if (profile != null && profile.id != null && profile.feature != null) {
                if (profile.userId == null || profile.userId.trim().isEmpty()) {
                    profile.userId = profile.id;
                }
                PROFILES.put(profile.id, profile);
            }
        }
    }

    private static void saveRegistry() throws IOException {
        RegistryFile registry = new RegistryFile();
        registry.profiles = new ArrayList<StoredProfile>(PROFILES.values());
        Files.write(REGISTRY_FILE, GSON.toJson(registry).getBytes(StandardCharsets.UTF_8));
    }

    private static JsonObject readJson(InputStream stream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
        StringBuilder builder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            builder.append(line);
        }
        return GSON.fromJson(builder.toString(), JsonObject.class);
    }

    private static String stringValue(JsonObject object, String key) {
        if (object == null || !object.has(key) || object.get(key).isJsonNull()) {
            return "";
        }
        return object.get(key).getAsString().trim();
    }

    private static String firstNonEmpty(String... values) {
        for (String value : values) {
            if (value != null && !value.trim().isEmpty()) {
                return value.trim();
            }
        }
        return "";
    }

    private static void addCors(HttpExchange exchange) {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        headers.set("Access-Control-Allow-Origin", "*");
        headers.set("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
        headers.set("Access-Control-Allow-Headers", "Content-Type");
    }

    private static void writeEmpty(HttpExchange exchange, int statusCode) throws IOException {
        exchange.sendResponseHeaders(statusCode, -1);
        exchange.close();
    }

    private static void writeJson(HttpExchange exchange, int statusCode, Object payload) throws IOException {
        byte[] body = GSON.toJson(payload).getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(statusCode, body.length);
        OutputStream outputStream = exchange.getResponseBody();
        outputStream.write(body);
        outputStream.close();
    }

    private static Map<String, Object> mapOf(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        for (int index = 0; index < pairs.length; index += 2) {
            result.put(String.valueOf(pairs[index]), pairs[index + 1]);
        }
        return result;
    }

    private static class RegistryFile {
        List<StoredProfile> profiles;
    }

    private static class StoredProfile {
        String id;
        String userId;
        String name;
        String role;
        String pin;
        float[] feature;

        Map<String, Object> toResponseMap() {
            return mapOf(
                    "id", id,
                    "userId", userId == null || userId.isEmpty() ? id : userId,
                    "name", name,
                    "role", role,
                    "pin", pin,
                    "backendManaged", true
            );
        }
    }
}
