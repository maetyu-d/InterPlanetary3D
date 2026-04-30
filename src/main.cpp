#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWindowWidth = 1600;
constexpr int kWindowHeight = 900;
constexpr int kChunkX = 24;
constexpr int kChunkY = 28;
constexpr int kChunkZ = 24;
constexpr int kWorldChunksX = 3;
constexpr int kWorldChunksZ = 3;
constexpr int kWorldX = kChunkX * kWorldChunksX;
constexpr int kWorldY = kChunkY;
constexpr int kWorldZ = kChunkZ * kWorldChunksZ;
constexpr int kChunkCount = kWorldChunksX * kWorldChunksZ;
constexpr float kBlockSize = 3.0f;
constexpr float kWalkSpeed = 5.4f * kBlockSize;
constexpr float kSprintMultiplier = 1.45f;
constexpr float kJumpVelocity = 7.4f * kBlockSize;
constexpr float kGravity = 20.0f * kBlockSize;
constexpr float kMouseSensitivity = 0.08f;
constexpr float kPlayerRadius = 0.32f * kBlockSize;
constexpr float kPlayerHeight = 1.75f * kBlockSize;
constexpr float kEyeHeight = 1.62f * kBlockSize;
constexpr float kReach = 6.5f * kBlockSize;
constexpr float kStep = 0.05f * kBlockSize;
constexpr float kCollisionInset = 0.001f;
constexpr float kForcefieldThickness = 0.10f * kBlockSize;
constexpr float kForcefieldOversize = 1.155f;
constexpr float kBombDropGravity = 17.0f * kBlockSize;
constexpr float kBombDropStep = 0.03f;
constexpr float kBombDropMaxTime = 12.0f;
constexpr float kSatelliteSize = 0.38f * kBlockSize;
constexpr float kSatelliteBeaconSize = 0.18f * kBlockSize;
constexpr float kSatelliteOrbitPadding = 9.0f * kBlockSize;
constexpr float kSatelliteOrbitPeriod = 28.0f;
constexpr float kSatelliteOrbitAdjustSpeed = 0.95f;
constexpr float kSatelliteOrbitSmoothing = 4.5f;
constexpr float kSatelliteOrbitSpeedAdjustRate = 0.85f;
constexpr float kSatelliteOrbitSpeedSmoothing = 4.0f;
constexpr float kSatelliteOrbitSpeedMin = 0.50f;
constexpr float kSatelliteOrbitSpeedMax = 1.00f;
constexpr float kSatelliteOrbitSlowAltitudeBoost = 8.0f * kBlockSize;
constexpr int kSatelliteOrbitSegments = 96;
constexpr float kSatelliteBlinkPeriod = 1.2f;
constexpr float kAtomicBombSize = 0.58f * kBlockSize;
constexpr float kAtomicBombExplosionDuration = 1.65f;
constexpr float kAtomicBombCraterRadius = 5.8f * kBlockSize;
constexpr float kAtomicBombBlastRadius = 10.5f * kBlockSize;
constexpr float kAtomicBombBounceDuration = 3.0f;
constexpr float kAtomicBombBounceHeight = 1.15f * kBlockSize;
constexpr float kAtomicBombBounceDrift = 2.4f * kBlockSize;
constexpr int kAtomicBombTrailCapacity = 14;
constexpr int kAtomicBombRingSegments = 40;
constexpr float kMissileSize = 0.45f * kBlockSize;
constexpr float kMissileOrbitPadding = 7.0f * kBlockSize;
constexpr float kMissileLaunchLift = 2.4f * kBlockSize;
constexpr float kMissileFinalDive = 2.0f * kBlockSize;
constexpr float kMissileAimRange = 6.0f * kBlockSize;
constexpr float kMissilePitchInfluence = 0.45f;
constexpr float kMissilePitchRangeInfluence = 0.85f;
constexpr float kMissileMinPower = 0.18f;
constexpr float kMissileMaxPower = 1.0f;
constexpr float kMissileChargeRate = 0.75f;
constexpr float kMissileDuration = 3.2f;
constexpr float kMissileExplosionDuration = 0.75f;
constexpr float kMissileExplosionRadius = 2.1f * kBlockSize;
constexpr int kMissileArcSegments = 96;
constexpr float kHandSwingDuration = 0.22f;
constexpr float kMiningSwingInterval = 0.16f;
constexpr float kPlacementSwingInterval = 0.18f;
constexpr float kCameraThumpDuration = 0.10f;
constexpr float kCameraSnapDuration = 0.16f;

enum BlockType : std::uint8_t {
  Air = 0,
  Crust = 1,
  DarkRock = 2,
  Ember = 3,
  Target = 4,
};

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 uv;
  float ao;
};

struct Player {
  glm::vec3 position{static_cast<float>(kWorldX) * kBlockSize * 0.5f, 16.0f * kBlockSize,
                     static_cast<float>(kWorldZ) * kBlockSize * 0.5f};
  glm::vec3 velocity{0.0f};
  float yaw = -90.0f;
  float pitch = -12.0f;
  bool onGround = false;
};

struct InputState {
  bool firstMouse = true;
  bool captureMouse = true;
  bool leftPressedLastFrame = false;
  bool rightPressedLastFrame = false;
  bool atomicBombPressedLastFrame = false;
  bool missilePressedLastFrame = false;
  bool launcherToggleLastFrame = false;
  bool jumpHeldLastFrame = false;
  double lastMouseX = kWindowWidth * 0.5;
  double lastMouseY = kWindowHeight * 0.5;
};

struct RaycastHit {
  glm::ivec3 block;
  glm::ivec3 previous;
  BlockType type = Air;
};

struct BombsitePrediction {
  glm::vec3 impactPos{0.0f};
  bool hitForcefield = false;
};

struct World {
  std::vector<BlockType> blocks;

  World() : blocks(kWorldX * kWorldY * kWorldZ, Air) {}

  bool inBounds(int x, int y, int z) const {
    return x >= 0 && y >= 0 && z >= 0 && x < kWorldX && y < kWorldY && z < kWorldZ;
  }

  BlockType get(int x, int y, int z) const {
    if (!inBounds(x, y, z)) {
      return Air;
    }
    return blocks[(y * kWorldZ + z) * kWorldX + x];
  }

  void set(int x, int y, int z, BlockType type) {
    if (!inBounds(x, y, z)) {
      return;
    }
    blocks[(y * kWorldZ + z) * kWorldX + x] = type;
  }
};

struct GLMesh {
  GLuint vao = 0;
  GLuint vbo = 0;
  GLsizei vertexCount = 0;
};

struct ChunkMesh {
  GLMesh mesh;
  bool dirty = true;
};

struct MissileState {
  bool active = false;
  bool exploding = false;
  bool hitTarget = false;
  glm::vec3 launchPos{0.0f};
  glm::vec3 startOrbitPos{0.0f};
  glm::vec3 endOrbitPos{0.0f};
  glm::vec3 impactPos{0.0f};
  glm::vec3 currentPos{0.0f};
  float progress = 0.0f;
  float duration = kMissileDuration;
  float explosionAge = 0.0f;
};

struct MissileAimState {
  bool charging = false;
  float power = 0.55f;
  float chargeDirection = 1.0f;
};

struct MissileSolution {
  glm::vec3 launchPos{0.0f};
  glm::vec3 startOrbitPos{0.0f};
  glm::vec3 endOrbitPos{0.0f};
  glm::vec3 impactPos{0.0f};
  float duration = kMissileDuration;
};

struct HandState {
  float swingTime = 0.0f;
  bool swinging = false;
};

struct MiningState {
  bool active = false;
  glm::ivec3 block{0};
  BlockType type = Air;
  float progress = 0.0f;
  float swingCooldown = 0.0f;
};

struct PlacementState {
  bool active = false;
  glm::ivec3 block{0};
  BlockType type = Air;
  float progress = 0.0f;
  float swingCooldown = 0.0f;
};

struct CameraFeedbackState {
  float thumpTime = 0.0f;
  float snapTime = 0.0f;
  bool thumping = false;
  bool snapping = false;
};

struct AtomicBombState {
  bool active = false;
  bool bouncing = false;
  bool exploding = false;
  bool hitForcefield = false;
  glm::vec3 position{0.0f};
  glm::vec3 velocity{0.0f};
  glm::vec3 impactPos{0.0f};
  glm::vec3 bounceNormal{0.0f, 1.0f, 0.0f};
  glm::vec3 bounceDrift{0.0f};
  float bounceAge = 0.0f;
  float explosionAge = 0.0f;
  std::array<glm::vec3, kAtomicBombTrailCapacity> trail{};
  int trailCount = 0;
};

struct AppState {
  Player player;
  glm::vec3 spawnPosition{0.0f};
  InputState input;
  World world;
  std::array<ChunkMesh, kChunkCount> chunkMeshes;
  std::optional<RaycastHit> hoveredBlock;
  BlockType selectedBlock = Crust;
  float satelliteOrbitYaw = 0.0f;
  float satelliteOrbitYawTarget = 0.0f;
  float satelliteOrbitPhase = 0.0f;
  float satelliteOrbitSpeed = 1.0f;
  float satelliteOrbitSpeedTarget = 1.0f;
  bool launcherEquipped = false;
  MissileAimState missileAim;
  MissileState missile;
  HandState hand;
  MiningState mining;
  PlacementState placing;
  CameraFeedbackState cameraFx;
  AtomicBombState atomicBomb;
};

AppState* gState = nullptr;

void triggerHandSwing(AppState& state);
void triggerCameraThump(AppState& state);
void triggerCameraSnap(AppState& state);
void resetMining(AppState& state);
void resetPlacement(AppState& state);
void dropAtomicBomb(AppState& state);
void updateAtomicBomb(AppState& state, float deltaTime);

int chunkIndexForCoords(int chunkX, int chunkZ) {
  return chunkZ * kWorldChunksX + chunkX;
}

float hashNoise(int x, int z) {
  std::uint32_t n = static_cast<std::uint32_t>(x * 374761393 + z * 668265263);
  n = (n ^ (n >> 13U)) * 1274126177U;
  n ^= n >> 16U;
  return static_cast<float>(n & 0xffffU) / 65535.0f;
}

float hashNoise3(int x, int y, int z) {
  std::uint32_t n = static_cast<std::uint32_t>(x * 374761393 + y * 1103515245 + z * 668265263);
  n = (n ^ (n >> 13U)) * 1274126177U;
  n ^= n >> 16U;
  return static_cast<float>(n & 0xffffU) / 65535.0f;
}

glm::vec3 eyePosition(const Player& player) {
  return player.position + glm::vec3(0.0f, kEyeHeight, 0.0f);
}

int worldToBlockCoord(float value) {
  return static_cast<int>(std::floor(value / kBlockSize));
}

glm::ivec3 worldToBlock(const glm::vec3& position) {
  return {worldToBlockCoord(position.x), worldToBlockCoord(position.y), worldToBlockCoord(position.z)};
}

glm::vec3 blockToWorld(const glm::ivec3& block) {
  return glm::vec3(block) * kBlockSize;
}

glm::vec3 blockCenterToWorld(const glm::ivec3& block) {
  return (glm::vec3(block) + glm::vec3(0.5f)) * kBlockSize;
}

glm::vec3 worldCenter() {
  return glm::vec3(static_cast<float>(kWorldX), static_cast<float>(kWorldY), static_cast<float>(kWorldZ)) *
         kBlockSize * 0.5f;
}

float satelliteOrbitRadius(float orbitSpeed) {
  const glm::vec3 halfExtents = glm::vec3(static_cast<float>(kWorldX), static_cast<float>(kWorldY),
                                          static_cast<float>(kWorldZ)) *
                                kBlockSize * 0.5f;
  const float slowAmount =
      std::clamp((kSatelliteOrbitSpeedMax - orbitSpeed) / (kSatelliteOrbitSpeedMax - kSatelliteOrbitSpeedMin),
                 0.0f, 1.0f);
  return glm::length(halfExtents) + kSatelliteOrbitPadding + slowAmount * kSatelliteOrbitSlowAltitudeBoost;
}

glm::vec3 satellitePositionAtAngle(float angle, float orbitYaw, float orbitSpeed) {
  const glm::vec3 center = worldCenter();
  const float orbitRadius = satelliteOrbitRadius(orbitSpeed);

  // Polar orbit: a vertical loop crossing above the top face and the opposite bottom face.
  const glm::vec3 localPos(0.0f, std::cos(angle) * orbitRadius, std::sin(angle) * orbitRadius);
  const float s = std::sin(orbitYaw);
  const float c = std::cos(orbitYaw);
  const glm::vec3 rotated(localPos.z * s, localPos.y, localPos.z * c);
  return center + rotated;
}

glm::vec3 satelliteTangentAtAngle(float angle, float orbitYaw, float orbitSpeed) {
  const float orbitRadius = satelliteOrbitRadius(orbitSpeed);
  const glm::vec3 localTangent(0.0f, -std::sin(angle) * orbitRadius, std::cos(angle) * orbitRadius);
  const float s = std::sin(orbitYaw);
  const float c = std::cos(orbitYaw);
  const glm::vec3 rotated(localTangent.z * s, localTangent.y, localTangent.z * c);
  return glm::normalize(rotated);
}

glm::vec3 targetCenter() {
  return glm::vec3((static_cast<float>(kWorldX) * 0.5f), 0.5f, (static_cast<float>(kWorldZ) * 0.5f)) * kBlockSize;
}

int topSolidYAt(const World& world, int x, int z) {
  for (int y = kWorldY - 1; y >= 0; --y) {
    if (world.get(x, y, z) != Air) {
      return y;
    }
  }
  return 0;
}

glm::vec3 findSpawnPosition(const World& world) {
  const int centerX = kWorldX / 2;
  const int centerZ = kWorldZ / 2;
  int bestX = centerX;
  int bestZ = centerZ;
  int bestY = topSolidYAt(world, centerX, centerZ);

  for (int radius = 0; radius < 8; ++radius) {
    for (int dz = -radius; dz <= radius; ++dz) {
      for (int dx = -radius; dx <= radius; ++dx) {
        const int x = std::clamp(centerX + dx, 0, kWorldX - 1);
        const int z = std::clamp(centerZ + dz, 0, kWorldZ - 1);
        const int y = topSolidYAt(world, x, z);
        if (y > bestY) {
          bestY = y;
          bestX = x;
          bestZ = z;
        }
      }
    }
  }

  return glm::vec3((static_cast<float>(bestX) + 0.5f) * kBlockSize,
                   (static_cast<float>(bestY) + 1.0f) * kBlockSize,
                   (static_cast<float>(bestZ) + 0.5f) * kBlockSize);
}

glm::vec3 cameraFront(const Player& player) {
  glm::vec3 front;
  front.x = std::cos(glm::radians(player.yaw)) * std::cos(glm::radians(player.pitch));
  front.y = std::sin(glm::radians(player.pitch));
  front.z = std::sin(glm::radians(player.yaw)) * std::cos(glm::radians(player.pitch));
  return glm::normalize(front);
}

glm::vec3 orbitSlerp(const glm::vec3& start, const glm::vec3& end, float t) {
  const float dotValue = std::clamp(glm::dot(start, end), -0.9999f, 0.9999f);
  const float theta = std::acos(dotValue);
  if (theta < 0.001f) {
    return glm::normalize(glm::mix(start, end, t));
  }

  const float sinTheta = std::sin(theta);
  const float a = std::sin((1.0f - t) * theta) / sinTheta;
  const float b = std::sin(t * theta) / sinTheta;
  return glm::normalize(start * a + end * b);
}

glm::vec3 quadraticBezier(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, float t) {
  const float inv = 1.0f - t;
  return inv * inv * a + 2.0f * inv * t * b + t * t * c;
}

bool sameBlock(const glm::ivec3& a, const glm::ivec3& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

float miningDurationFor(BlockType type) {
  switch (type) {
    case Crust: return 0.35f;
    case Ember: return 0.55f;
    case DarkRock: return 0.85f;
    case Target: return 1000.0f;
    case Air: return 0.0f;
  }
  return 0.5f;
}

float placementDurationFor(BlockType type) {
  switch (type) {
    case Crust: return 0.78f;
    case Ember: return 1.02f;
    case DarkRock: return 1.26f;
    case Target: return 1000.0f;
    case Air: return 0.0f;
  }
  return 0.90f;
}

MissileSolution buildMissileSolution(const AppState& state, std::optional<float> powerOverride = std::nullopt) {
  MissileSolution solution;
  const glm::vec3 center = worldCenter();
  const glm::vec3 outward = glm::normalize(state.player.position - center);
  const glm::vec3 front = cameraFront(state.player);
  const float power = std::clamp(powerOverride.value_or(state.missileAim.power), kMissileMinPower, kMissileMaxPower);

  glm::vec3 surfaceForward = front - outward * glm::dot(front, outward);
  if (glm::dot(surfaceForward, surfaceForward) > 0.0001f) {
    surfaceForward = glm::normalize(surfaceForward);
  } else {
    surfaceForward = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), outward));
    if (glm::dot(surfaceForward, surfaceForward) <= 0.0001f) {
      surfaceForward = glm::vec3(1.0f, 0.0f, 0.0f);
    }
  }

  const float pitchBias = std::clamp(state.player.pitch / 75.0f, -1.0f, 1.0f);
  glm::vec3 aimOffset(surfaceForward.x, 0.0f, surfaceForward.z);
  if (glm::dot(aimOffset, aimOffset) > 0.0001f) {
    const float pitchRangeScale = std::clamp(1.0f + pitchBias * kMissilePitchRangeInfluence, 0.15f, 1.85f);
    aimOffset = glm::normalize(aimOffset) * (kMissileAimRange * pitchRangeScale * power);
  } else {
    aimOffset = glm::vec3(0.0f);
  }

  const float radialWeight = 0.44f + power * 0.36f + pitchBias * kMissilePitchInfluence;
  const float tangentWeight = 1.02f - power * 0.22f - pitchBias * 0.18f;
  const glm::vec3 launchDir = glm::normalize(surfaceForward * tangentWeight + outward * radialWeight);

  solution.launchPos =
      eyePosition(state.player) + outward * (kMissileLaunchLift + power * 0.5f * kBlockSize) +
      launchDir * ((0.75f + power * 0.5f) * kBlockSize);

  solution.impactPos = glm::vec3(
      std::clamp(state.player.position.x + aimOffset.x, 0.5f * kBlockSize,
                 static_cast<float>(kWorldX) * kBlockSize - 0.5f * kBlockSize),
      0.5f * kBlockSize,
      std::clamp(state.player.position.z + aimOffset.z, 0.5f * kBlockSize,
                 static_cast<float>(kWorldZ) * kBlockSize - 0.5f * kBlockSize));

  const glm::vec3 startDir = launchDir;
  const glm::vec3 endDir = glm::normalize(solution.impactPos - center);
  const glm::vec3 halfExtents = glm::vec3(static_cast<float>(kWorldX), static_cast<float>(kWorldY),
                                          static_cast<float>(kWorldZ)) *
                                kBlockSize * 0.5f;
  const float orbitRadius = glm::length(halfExtents) + kMissileOrbitPadding + power * 2.2f * kBlockSize;

  solution.startOrbitPos = center + startDir * orbitRadius;
  solution.endOrbitPos = center + endDir * orbitRadius;
  solution.duration = glm::mix(kMissileDuration * 0.78f, kMissileDuration * 1.18f, power);
  return solution;
}

glm::vec3 missilePositionAt(const MissileSolution& solution, float t) {
  const glm::vec3 center = worldCenter();
  const glm::vec3 startDir = glm::normalize(solution.startOrbitPos - center);
  const glm::vec3 endDir = glm::normalize(solution.endOrbitPos - center);
  const float orbitRadius = glm::length(solution.startOrbitPos - center);
  if (t < 0.18f) {
    const float s = glm::smoothstep(0.0f, 1.0f, t / 0.18f);
    const glm::vec3 launchControl =
        solution.launchPos + startDir * (orbitRadius - glm::length(solution.launchPos - center)) * 0.55f;
    return quadraticBezier(solution.launchPos, launchControl, solution.startOrbitPos, s);
  }
  if (t < 0.84f) {
    const float s = glm::smoothstep(0.0f, 1.0f, (t - 0.18f) / 0.66f);
    const glm::vec3 orbitDir = orbitSlerp(startDir, endDir, s);
    return center + orbitDir * orbitRadius;
  }

  const float s = glm::smoothstep(0.0f, 1.0f, (t - 0.84f) / 0.16f);
  const glm::vec3 diveTarget = solution.impactPos + glm::vec3(0.0f, kMissileFinalDive, 0.0f);
  const glm::vec3 diveControl = glm::mix(solution.endOrbitPos, diveTarget, 0.52f) - endDir * (1.2f * kBlockSize);
  return quadraticBezier(solution.endOrbitPos, diveControl, diveTarget, s);
}

GLuint compileShader(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (success == GL_TRUE) {
    return shader;
  }

  GLint length = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
  std::string log(length, '\0');
  glGetShaderInfoLog(shader, length, nullptr, log.data());
  glDeleteShader(shader);
  throw std::runtime_error("Shader compilation failed: " + log);
}

GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader) {
  const GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);

  GLint success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (success == GL_TRUE) {
    return program;
  }

  GLint length = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
  std::string log(length, '\0');
  glGetProgramInfoLog(program, length, nullptr, log.data());
  glDeleteProgram(program);
  throw std::runtime_error("Program linking failed: " + log);
}

GLuint createWorldProgram() {
  static constexpr const char* kVertexShader = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aUv;
    layout (location = 3) in float aAo;

    out vec3 vWorldPos;
    out vec3 vNormal;
    out vec2 vUv;
    out float vAo;

    uniform mat4 uModel;
    uniform mat4 uView;
    uniform mat4 uProjection;

    void main() {
      vec4 worldPos = uModel * vec4(aPos, 1.0);
      vWorldPos = worldPos.xyz;
      vNormal = mat3(transpose(inverse(uModel))) * aNormal;
      vUv = aUv;
      vAo = aAo;
      gl_Position = uProjection * uView * worldPos;
    }
  )";

  static constexpr const char* kFragmentShader = R"(
    #version 330 core
    in vec3 vWorldPos;
    in vec3 vNormal;
    in vec2 vUv;
    in float vAo;

    out vec4 FragColor;

    uniform sampler2D uAtlas;
    uniform vec3 uCameraPos;
    uniform vec3 uWorldCenter;
    uniform vec3 uFogColor;
    uniform float uExposure;
    uniform float uFogDensity;

    void main() {
      vec3 tex = texture(uAtlas, vUv).rgb;
      vec3 normal = normalize(vNormal);
      vec3 surfaceUp = normalize(vWorldPos - uWorldCenter);

      vec3 lightDir = normalize(vec3(-0.42, 0.98, -0.18));
      float diffuse = max(dot(normal, normalize(lightDir + surfaceUp * 0.85)), 0.0);
      float viewDot = max(dot(normal, normalize(uCameraPos - vWorldPos)), 0.0);
      float rim = pow(1.0 - viewDot, 3.0) * 0.18;
      float shadowBand = smoothstep(0.16, 0.82, diffuse);
      float lowLight = 0.18 + shadowBand * 0.68 + rim;
      float radialTop = smoothstep(0.12, 1.0, dot(normal, surfaceUp));
      float radialBottom = smoothstep(0.12, 1.0, dot(normal, -surfaceUp));
      float topBias = radialTop * 0.30;
      float bottomBias = radialBottom * -0.08;
      float sideBias = (1.0 - abs(dot(normal, surfaceUp))) * -0.03;
      lowLight += topBias + bottomBias + sideBias;

      float emberGlow = smoothstep(0.24, 0.42, tex.r - tex.g) * 0.26;
      float radialDistance = length(vWorldPos - uWorldCenter);
      float verticalHeat = smoothstep(10.0, 28.0, radialDistance) * (0.08 + topBias * 0.34);
      vec3 lit = tex * (lowLight + emberGlow + verticalHeat);
      lit *= vAo;
      lit = mix(lit * 0.56, lit * 1.52, shadowBand);
      lit = pow(max(lit, vec3(0.0)), vec3(0.82));

      float lowerHalf = smoothstep(-0.75, 0.75, uWorldCenter.y - vWorldPos.y);
      vec3 upperTint = vec3(1.00, 0.82, 0.82);
      vec3 lowerTint = vec3(0.18, 0.34, 1.32);
      lit *= mix(upperTint, lowerTint, lowerHalf);

      float dist = distance(uCameraPos, vWorldPos);
      float groundFog = smoothstep(13.0, -6.0, vWorldPos.y) * 0.12;
      float fogFactor = clamp(1.0 - exp(-(dist * uFogDensity + groundFog)), 0.0, 1.0);
      vec3 lowerFogColor = vec3(0.05, 0.10, 0.24);
      vec3 fogColor = mix(uFogColor, lowerFogColor, lowerHalf);
      vec3 color = mix(lit, fogColor, fogFactor);
      color = mix(color * 0.72, color * 1.16, shadowBand);

      FragColor = vec4(color * uExposure, 1.0);
    }
  )";

  const GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
  const GLuint program = linkProgram(vs, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

GLuint createSkyProgram() {
  static constexpr const char* kVertexShader = R"(
    #version 330 core
    const vec2 positions[3] = vec2[](
      vec2(-1.0, -1.0),
      vec2( 3.0, -1.0),
      vec2(-1.0,  3.0)
    );

    out vec2 vUv;

    void main() {
      vec2 pos = positions[gl_VertexID];
      vUv = pos * 0.5 + 0.5;
      gl_Position = vec4(pos, 0.0, 1.0);
    }
  )";

  static constexpr const char* kFragmentShader = R"(
    #version 330 core
    in vec2 vUv;

    out vec4 FragColor;

    uniform float uTime;

    void main() {
      vec3 top = vec3(0.18, 0.03, 0.02);
      vec3 horizon = vec3(0.58, 0.16, 0.08);
      vec3 pit = vec3(0.03, 0.00, 0.00);

      float horizonBand = smoothstep(0.10, 0.62, vUv.y);
      vec3 color = mix(pit, horizon, horizonBand);
      color = mix(color, top, smoothstep(0.58, 1.0, vUv.y));

      float heatWave = sin(vUv.x * 19.0 + uTime * 0.3) * 0.01 + sin(vUv.x * 7.0 - uTime * 0.15) * 0.015;
      color += vec3(0.18, 0.05, 0.01) * smoothstep(0.15, 0.55, vUv.y + heatWave) * (1.0 - smoothstep(0.55, 0.9, vUv.y));
      color = pow(color, vec3(0.82));

      FragColor = vec4(color, 1.0);
    }
  )";

  const GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
  const GLuint program = linkProgram(vs, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

GLuint createOutlineProgram() {
  static constexpr const char* kVertexShader = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;

    uniform mat4 uModel;
    uniform mat4 uView;
    uniform mat4 uProjection;

    void main() {
      gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
    }
  )";

  static constexpr const char* kFragmentShader = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 uColor;

    void main() {
      FragColor = vec4(uColor, 1.0);
    }
  )";

  const GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
  const GLuint program = linkProgram(vs, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

GLuint createColorProgram() {
  static constexpr const char* kVertexShader = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;

    uniform mat4 uModel;
    uniform mat4 uView;
    uniform mat4 uProjection;

    void main() {
      gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
    }
  )";

  static constexpr const char* kFragmentShader = R"(
    #version 330 core
    out vec4 FragColor;

    uniform vec3 uColor;

    void main() {
      FragColor = vec4(uColor, 1.0);
    }
  )";

  const GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
  const GLuint program = linkProgram(vs, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

std::vector<std::uint8_t> generateAtlas(int tileSize, int gridSize) {
  const int width = tileSize * gridSize;
  const int height = tileSize * 2;
  std::vector<std::uint8_t> pixels(width * height * 3, 0);

  auto writeTile = [&](int tileX, int tileY, glm::u8vec3 base, glm::u8vec3 alt, bool glowing) {
    for (int y = 0; y < tileSize; ++y) {
      for (int x = 0; x < tileSize; ++x) {
        const int px = tileX * tileSize + x;
        const int py = tileY * tileSize + y;
        const int index = (py * width + px) * 3;
        const int pattern = ((x / 3) ^ (y / 3) ^ ((x * 7 + y * 3) / 5)) & 1;

        glm::u8vec3 color = pattern == 0 ? base : alt;
        if (glowing && ((x + y) % 11 == 0 || (x * y) % 29 == 0)) {
          color.r = static_cast<std::uint8_t>(std::min(255, color.r + 55));
          color.g = static_cast<std::uint8_t>(std::min(255, color.g + 18));
        }

        pixels[index + 0] = color.r;
        pixels[index + 1] = color.g;
        pixels[index + 2] = color.b;
      }
    }
  };

  writeTile(0, 0, {78, 15, 8}, {46, 6, 3}, false);
  writeTile(1, 0, {58, 9, 8}, {24, 5, 5}, false);
  writeTile(0, 1, {112, 31, 14}, {64, 12, 7}, true);
  writeTile(1, 1, {95, 22, 16}, {40, 7, 5}, false);
  writeTile(2, 0, {214, 182, 38}, {126, 96, 12}, false);
  return pixels;
}

GLuint createAtlasTexture() {
  constexpr int tileSize = 16;
  constexpr int gridSize = 3;
  const auto pixels = generateAtlas(tileSize, gridSize);

  GLuint texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, tileSize * gridSize, tileSize * 2, 0, GL_RGB,
               GL_UNSIGNED_BYTE, pixels.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  return texture;
}

glm::vec2 atlasUv(int tileIndex, int corner) {
  constexpr float columns = 3.0f;
  constexpr float rows = 2.0f;
  const float tileU = 1.0f / columns;
  const float tileV = 1.0f / rows;
  const int tileX = tileIndex % 3;
  const int tileY = tileIndex / 3;
  const float u0 = tileX * tileU;
  const float v0 = tileY * tileV;
  const float u1 = u0 + tileU;
  const float v1 = v0 + tileV;

  switch (corner) {
    case 0: return {u0, v0};
    case 1: return {u1, v0};
    case 2: return {u1, v1};
    case 3: return {u0, v1};
    default: return {u0, v0};
  }
}

int textureIndexFor(BlockType type) {
  switch (type) {
    case Crust: return 0;
    case DarkRock: return 1;
    case Ember: return 2;
    case Target: return 4;
    case Air: return 3;
  }
  return 0;
}

void markChunkDirty(AppState& state, int chunkX, int chunkZ) {
  if (chunkX < 0 || chunkZ < 0 || chunkX >= kWorldChunksX || chunkZ >= kWorldChunksZ) {
    return;
  }
  state.chunkMeshes[chunkIndexForCoords(chunkX, chunkZ)].dirty = true;
}

void markDirtyAroundBlock(AppState& state, int x, int z) {
  const int chunkX = x / kChunkX;
  const int chunkZ = z / kChunkZ;
  markChunkDirty(state, chunkX, chunkZ);

  if (x % kChunkX == 0) markChunkDirty(state, chunkX - 1, chunkZ);
  if (x % kChunkX == kChunkX - 1) markChunkDirty(state, chunkX + 1, chunkZ);
  if (z % kChunkZ == 0) markChunkDirty(state, chunkX, chunkZ - 1);
  if (z % kChunkZ == kChunkZ - 1) markChunkDirty(state, chunkX, chunkZ + 1);
}

void markAllChunksDirty(AppState& state) {
  for (ChunkMesh& chunk : state.chunkMeshes) {
    chunk.dirty = true;
  }
}

void generateWorld(World& world) {
  struct FaceSample {
    int inset = 0;
    float scar = 0.0f;
  };

  auto sampleFace = [&](float na, float nb, int seedA, int seedB, int maxInset) -> FaceSample {
    const float broadShape =
        std::sin((na + seedA * 0.071f) * 3.1f) * 1.5f + std::cos((nb + seedB * 0.053f) * 2.7f) * 1.4f;
    const float ridgeA =
        1.0f - std::abs(std::sin(((na + seedA * 0.03f) * 0.78f + (nb + seedB * 0.05f) * 1.18f) * 7.5f));
    const float ridgeB =
        1.0f - std::abs(std::sin(((na + seedA * 0.02f) * 1.35f - (nb + seedB * 0.04f) * 0.62f) * 5.8f + 0.8f));
    const float ridgeShape = std::pow(std::max(ridgeA, ridgeB), 3.2f) * 2.8f;
    const float basin =
        -std::exp(-std::pow((na - 0.50f) * 3.7f, 2.0f) - std::pow((nb - 0.52f) * 3.7f, 2.0f)) * 1.1f;
    const float plateauMask =
        std::pow(std::max(0.0f, std::sin((na + seedA * 0.09f) * 2.2f + 0.7f) *
                                     std::cos((nb + seedB * 0.08f) * 2.0f - 0.4f)),
                 2.0f);
    const float plateauLift = plateauMask * 1.8f;
    const float noise = (hashNoise(static_cast<int>(na * 997.0f) + seedA * 37,
                                   static_cast<int>(nb * 991.0f) + seedB * 41) -
                         0.5f) *
                        1.0f;

    float shapedInset = 4.2f + broadShape + ridgeShape + basin + plateauLift + noise;
    shapedInset = std::round(shapedInset);

    const float scarLineA =
        1.0f - std::abs(std::sin(((na + seedA * 0.05f) * 2.4f - (nb + seedB * 0.03f) * 0.9f) * 9.5f + 0.3f));
    const float scarLineB =
        1.0f - std::abs(std::sin(((na + seedA * 0.04f) * 1.1f + (nb + seedB * 0.07f) * 2.2f) * 8.2f - 1.1f));
    const float lavaScar =
        std::pow(std::max(scarLineA, scarLineB), 8.0f) *
        (0.55f + hashNoise(static_cast<int>(na * 751.0f) + seedA * 17,
                           static_cast<int>(nb * 743.0f) + seedB * 29) *
                    0.7f);

    FaceSample sample;
    sample.inset = std::clamp(static_cast<int>(shapedInset), 2, maxInset);
    sample.scar = lavaScar;
    return sample;
  };

  auto idxXZ = [](int x, int z) { return z * kWorldX + x; };
  auto idxYZ = [](int y, int z) { return z * kWorldY + y; };
  auto idxXY = [](int x, int y) { return y * kWorldX + x; };

  std::vector<FaceSample> topFace(kWorldX * kWorldZ);
  std::vector<FaceSample> bottomFace(kWorldX * kWorldZ);
  std::vector<FaceSample> leftFace(kWorldY * kWorldZ);
  std::vector<FaceSample> rightFace(kWorldY * kWorldZ);
  std::vector<FaceSample> frontFace(kWorldX * kWorldY);
  std::vector<FaceSample> backFace(kWorldX * kWorldY);

  const int maxYInset = std::max(3, kWorldY / 3);
  const int maxXInset = std::max(4, kWorldX / 7);
  const int maxZInset = std::max(4, kWorldZ / 7);

  for (int z = 0; z < kWorldZ; ++z) {
    for (int x = 0; x < kWorldX; ++x) {
      const float nx = static_cast<float>(x) / static_cast<float>(kWorldX);
      const float nz = static_cast<float>(z) / static_cast<float>(kWorldZ);
      topFace[idxXZ(x, z)] = sampleFace(nx, nz, 3, 11, maxYInset);
      bottomFace[idxXZ(x, z)] = sampleFace(nx, nz, 17, 23, maxYInset);
    }
  }

  for (int z = 0; z < kWorldZ; ++z) {
    for (int y = 0; y < kWorldY; ++y) {
      const float ny = static_cast<float>(y) / static_cast<float>(kWorldY);
      const float nz = static_cast<float>(z) / static_cast<float>(kWorldZ);
      leftFace[idxYZ(y, z)] = sampleFace(ny, nz, 29, 37, maxXInset);
      rightFace[idxYZ(y, z)] = sampleFace(ny, nz, 41, 47, maxXInset);
    }
  }

  for (int y = 0; y < kWorldY; ++y) {
    for (int x = 0; x < kWorldX; ++x) {
      const float nx = static_cast<float>(x) / static_cast<float>(kWorldX);
      const float ny = static_cast<float>(y) / static_cast<float>(kWorldY);
      frontFace[idxXY(x, y)] = sampleFace(nx, ny, 53, 61, maxZInset);
      backFace[idxXY(x, y)] = sampleFace(nx, ny, 71, 79, maxZInset);
    }
  }

  for (int z = 0; z < kWorldZ; ++z) {
    for (int y = 0; y < kWorldY; ++y) {
      for (int x = 0; x < kWorldX; ++x) {
        const FaceSample top = topFace[idxXZ(x, z)];
        const FaceSample bottom = bottomFace[idxXZ(x, z)];
        const FaceSample left = leftFace[idxYZ(y, z)];
        const FaceSample right = rightFace[idxYZ(y, z)];
        const FaceSample front = frontFace[idxXY(x, y)];
        const FaceSample back = backFace[idxXY(x, y)];

        const int topLimit = (kWorldY - 1) - top.inset;
        const int bottomLimit = bottom.inset;
        const int leftLimit = left.inset;
        const int rightLimit = (kWorldX - 1) - right.inset;
        const int frontLimit = front.inset;
        const int backLimit = (kWorldZ - 1) - back.inset;

        if (x < leftLimit || x > rightLimit ||
            y < bottomLimit || y > topLimit ||
            z < frontLimit || z > backLimit) {
          continue;
        }

        const int depthTop = topLimit - y;
        const int depthBottom = y - bottomLimit;
        const int depthLeft = x - leftLimit;
        const int depthRight = rightLimit - x;
        const int depthFront = z - frontLimit;
        const int depthBack = backLimit - z;
        const int surfaceDepth = std::min({depthTop, depthBottom, depthLeft, depthRight, depthFront, depthBack});
        const float lavaScar = std::max({top.scar, bottom.scar, left.scar, right.scar, front.scar, back.scar});
        const bool upperHalf = y >= (kWorldY / 2);

        BlockType type = Crust;
        if (surfaceDepth > 4) {
          type = DarkRock;
        } else {
          const bool nearSurface = surfaceDepth < 2;
          const float emberMask =
              hashNoise3(x + 31, y * 2 + 11, z + 17) + hashNoise(x + y, z - y) * 0.28f + lavaScar;
          if (nearSurface && !upperHalf && emberMask > 1.08f) {
            type = Ember;
          } else if (nearSurface && lavaScar > 0.72f) {
            type = DarkRock;
          }
        }

        world.set(x, y, z, type);
      }
    }
  }
}

std::vector<Vertex> buildChunkMesh(const World& world, int chunkX, int chunkZ) {
  std::vector<Vertex> vertices;
  vertices.reserve(kChunkX * kChunkY * kChunkZ * 6);

  const int minX = chunkX * kChunkX;
  const int minZ = chunkZ * kChunkZ;
  const int maxX = minX + kChunkX;
  const int maxZ = minZ + kChunkZ;

  const std::array<glm::ivec3, 6> offsets = {
      glm::ivec3{0, 0, -1}, glm::ivec3{0, 0, 1}, glm::ivec3{-1, 0, 0},
      glm::ivec3{1, 0, 0},  glm::ivec3{0, 1, 0}, glm::ivec3{0, -1, 0},
  };

  const std::array<std::array<glm::vec3, 4>, 6> corners = {{
      {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}},
      {{{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}}},
      {{{0, 0, 1}, {0, 0, 0}, {0, 1, 0}, {0, 1, 1}}},
      {{{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}}},
      {{{0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}}},
      {{{0, 0, 1}, {1, 0, 1}, {1, 0, 0}, {0, 0, 0}}},
  }};

  const std::array<glm::vec3, 6> normals = {
      glm::vec3{0, 0, -1}, glm::vec3{0, 0, 1}, glm::vec3{-1, 0, 0},
      glm::vec3{1, 0, 0},  glm::vec3{0, 1, 0}, glm::vec3{0, -1, 0},
  };

  auto solid = [&](int sx, int sy, int sz) -> bool {
    return world.get(sx, sy, sz) != Air;
  };

  auto aoValue = [&](bool side1, bool side2, bool corner) -> float {
    const int occlusion = side1 && side2 ? 3 : static_cast<int>(side1) + static_cast<int>(side2) + static_cast<int>(corner);
    return 1.0f - static_cast<float>(occlusion) * 0.14f;
  };

  auto faceAo = [&](int x, int y, int z, int face, int cornerIndex) -> float {
    switch (face) {
      case 0: {
        const int sx = cornerIndex == 0 || cornerIndex == 3 ? -1 : 1;
        const int sy = cornerIndex <= 1 ? -1 : 1;
        return aoValue(solid(x + sx, y, z), solid(x, y + sy, z), solid(x + sx, y + sy, z));
      }
      case 1: {
        const int sx = cornerIndex == 1 || cornerIndex == 2 ? -1 : 1;
        const int sy = cornerIndex <= 1 ? -1 : 1;
        return aoValue(solid(x + sx, y, z), solid(x, y + sy, z), solid(x + sx, y + sy, z));
      }
      case 2: {
        const int sz = cornerIndex == 0 || cornerIndex == 3 ? 1 : -1;
        const int sy = cornerIndex <= 1 ? -1 : 1;
        return aoValue(solid(x, y, z + sz), solid(x, y + sy, z), solid(x, y + sy, z + sz));
      }
      case 3: {
        const int sz = cornerIndex == 0 || cornerIndex == 3 ? -1 : 1;
        const int sy = cornerIndex <= 1 ? -1 : 1;
        return aoValue(solid(x, y, z + sz), solid(x, y + sy, z), solid(x, y + sy, z + sz));
      }
      case 4: {
        const int sx = cornerIndex == 0 || cornerIndex == 3 ? -1 : 1;
        const int sz = cornerIndex <= 1 ? -1 : 1;
        return aoValue(solid(x + sx, y, z), solid(x, y, z + sz), solid(x + sx, y, z + sz));
      }
      case 5: {
        const int sx = cornerIndex == 0 || cornerIndex == 3 ? -1 : 1;
        const int sz = cornerIndex <= 1 ? 1 : -1;
        return aoValue(solid(x + sx, y, z), solid(x, y, z + sz), solid(x + sx, y, z + sz));
      }
      default:
        return 1.0f;
    }
  };

  for (int y = 0; y < kWorldY; ++y) {
    for (int z = minZ; z < maxZ; ++z) {
      for (int x = minX; x < maxX; ++x) {
        const BlockType block = world.get(x, y, z);
        if (block == Air) {
          continue;
        }

        const int tile = textureIndexFor(block);
        for (int face = 0; face < 6; ++face) {
          const glm::ivec3 n = offsets[face];
          if (world.get(x + n.x, y + n.y, z + n.z) != Air) {
            continue;
          }

          const glm::vec3 base = glm::vec3(x, y, z) * kBlockSize;
          const auto uv0 = atlasUv(tile, 0);
          const auto uv1 = atlasUv(tile, 1);
          const auto uv2 = atlasUv(tile, 2);
          const auto uv3 = atlasUv(tile, 3);
          const float ao0 = faceAo(x, y, z, face, 0);
          const float ao1 = faceAo(x, y, z, face, 1);
          const float ao2 = faceAo(x, y, z, face, 2);
          const float ao3 = faceAo(x, y, z, face, 3);

          vertices.push_back({base + corners[face][0] * kBlockSize, normals[face], uv0, ao0});
          vertices.push_back({base + corners[face][2] * kBlockSize, normals[face], uv2, ao2});
          vertices.push_back({base + corners[face][1] * kBlockSize, normals[face], uv1, ao1});
          vertices.push_back({base + corners[face][2] * kBlockSize, normals[face], uv2, ao2});
          vertices.push_back({base + corners[face][0] * kBlockSize, normals[face], uv0, ao0});
          vertices.push_back({base + corners[face][3] * kBlockSize, normals[face], uv3, ao3});
        }
      }
    }
  }

  return vertices;
}

void uploadMesh(GLMesh& mesh, const std::vector<Vertex>& vertices) {
  if (mesh.vao == 0) {
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
  }

  glBindVertexArray(mesh.vao);
  glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(),
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, uv)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, ao)));
  glEnableVertexAttribArray(3);

  mesh.vertexCount = static_cast<GLsizei>(vertices.size());
}

void rebuildDirtyChunks(AppState& state) {
  for (int chunkZ = 0; chunkZ < kWorldChunksZ; ++chunkZ) {
    for (int chunkX = 0; chunkX < kWorldChunksX; ++chunkX) {
      ChunkMesh& chunk = state.chunkMeshes[chunkIndexForCoords(chunkX, chunkZ)];
      if (!chunk.dirty) {
        continue;
      }
      const auto vertices = buildChunkMesh(state.world, chunkX, chunkZ);
      uploadMesh(chunk.mesh, vertices);
      chunk.dirty = false;
    }
  }
}

void clearExplosionCrater(AppState& state, const glm::vec3& impactPos, float radius) {
  const glm::ivec3 center = worldToBlock(impactPos);
  const int blockRadius = static_cast<int>(std::ceil(radius / kBlockSize));
  const float radiusSq = radius * radius;

  for (int z = center.z - blockRadius; z <= center.z + blockRadius; ++z) {
    for (int y = center.y - blockRadius; y <= center.y + blockRadius; ++y) {
      for (int x = center.x - blockRadius; x <= center.x + blockRadius; ++x) {
        if (!state.world.inBounds(x, y, z)) {
          continue;
        }
        const glm::vec3 cellCenter = blockCenterToWorld({x, y, z});
        if (glm::dot(cellCenter - impactPos, cellCenter - impactPos) > radiusSq) {
          continue;
        }
        if (state.world.get(x, y, z) == Target) {
          continue;
        }
        state.world.set(x, y, z, Air);
        markDirtyAroundBlock(state, x, z);
      }
    }
  }
}

void pushAtomicBombTrail(AtomicBombState& bomb, const glm::vec3& position) {
  const int writeCount = std::min(bomb.trailCount + 1, kAtomicBombTrailCapacity);
  for (int i = writeCount - 1; i > 0; --i) {
    bomb.trail[static_cast<std::size_t>(i)] = bomb.trail[static_cast<std::size_t>(i - 1)];
  }
  bomb.trail[0] = position;
  bomb.trailCount = writeCount;
}

void startAtomicBombExplosion(AppState& state, const glm::vec3& impactPos, bool hitForcefield) {
  state.atomicBomb.active = false;
  state.atomicBomb.bouncing = false;
  state.atomicBomb.exploding = true;
  state.atomicBomb.hitForcefield = hitForcefield;
  state.atomicBomb.impactPos = impactPos;
  state.atomicBomb.position = impactPos;
  state.atomicBomb.explosionAge = 0.0f;
  pushAtomicBombTrail(state.atomicBomb, impactPos);
  triggerCameraThump(state);
  triggerCameraSnap(state);
  if (!hitForcefield) {
    clearExplosionCrater(state, impactPos, kAtomicBombCraterRadius);
  }
}

void startAtomicBombBounce(AppState& state, const glm::vec3& impactPos, bool hitForcefield,
                           const glm::vec3& incomingVelocity) {
  state.atomicBomb.active = false;
  state.atomicBomb.bouncing = true;
  state.atomicBomb.exploding = false;
  state.atomicBomb.hitForcefield = hitForcefield;
  state.atomicBomb.impactPos = impactPos;
  state.atomicBomb.position = impactPos;
  state.atomicBomb.bounceAge = 0.0f;
  state.atomicBomb.explosionAge = 0.0f;
  state.atomicBomb.trailCount = 0;
  const glm::vec3 center = worldCenter();
  state.atomicBomb.bounceNormal =
      hitForcefield ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::normalize(impactPos - center);
  glm::vec3 lateral = incomingVelocity - state.atomicBomb.bounceNormal * glm::dot(incomingVelocity, state.atomicBomb.bounceNormal);
  if (glm::dot(lateral, lateral) > 0.0001f) {
    lateral = glm::normalize(lateral);
  } else {
    lateral = glm::normalize(glm::cross(state.atomicBomb.bounceNormal, glm::vec3(0.0f, 0.0f, 1.0f)));
    if (glm::dot(lateral, lateral) <= 0.0001f) {
      lateral = glm::vec3(1.0f, 0.0f, 0.0f);
    }
  }
  state.atomicBomb.bounceDrift = lateral;
  pushAtomicBombTrail(state.atomicBomb, impactPos);
  triggerCameraThump(state);
}

bool isSolid(const World& world, int x, int y, int z) {
  return world.get(x, y, z) != Air;
}

bool collidesAt(const World& world, const glm::vec3& position) {
  const glm::vec3 min(position.x - kPlayerRadius + kCollisionInset, position.y + kCollisionInset,
                      position.z - kPlayerRadius + kCollisionInset);
  const glm::vec3 max(position.x + kPlayerRadius - kCollisionInset, position.y + kPlayerHeight - kCollisionInset,
                      position.z + kPlayerRadius - kCollisionInset);

  const int minX = worldToBlockCoord(min.x);
  const int maxX = worldToBlockCoord(max.x);
  const int minY = worldToBlockCoord(min.y);
  const int maxY = worldToBlockCoord(max.y);
  const int minZ = worldToBlockCoord(min.z);
  const int maxZ = worldToBlockCoord(max.z);

  for (int y = minY; y <= maxY; ++y) {
    for (int z = minZ; z <= maxZ; ++z) {
      for (int x = minX; x <= maxX; ++x) {
        if (isSolid(world, x, y, z)) {
          return true;
        }
      }
    }
  }

  return false;
}

std::optional<RaycastHit> raycast(const World& world, const glm::vec3& origin, const glm::vec3& direction, float maxDistance) {
  glm::vec3 pos = origin;
  glm::ivec3 previous = worldToBlock(origin);

  for (float travelled = 0.0f; travelled <= maxDistance; travelled += kStep) {
    pos = origin + direction * travelled;
    const glm::ivec3 block = worldToBlock(pos);
    const BlockType type = world.get(block.x, block.y, block.z);
    if (type != Air) {
      return RaycastHit{block, previous, type};
    }
    previous = block;
  }

  return std::nullopt;
}

bool playerIntersectsBlock(const glm::vec3& position, const glm::ivec3& block) {
  const glm::vec3 minA(position.x - kPlayerRadius + kCollisionInset, position.y + kCollisionInset,
                       position.z - kPlayerRadius + kCollisionInset);
  const glm::vec3 maxA(position.x + kPlayerRadius - kCollisionInset, position.y + kPlayerHeight - kCollisionInset,
                       position.z + kPlayerRadius - kCollisionInset);

  const glm::vec3 minB = blockToWorld(block);
  const glm::vec3 maxB = minB + glm::vec3(kBlockSize);

  return minA.x < maxB.x && maxA.x > minB.x &&
         minA.y < maxB.y && maxA.y > minB.y &&
         minA.z < maxB.z && maxA.z > minB.z;
}

bool playerTouchesForcefield(const glm::vec3& position) {
  const float playerMinY = position.y + kCollisionInset;
  const float playerMaxY = position.y + kPlayerHeight - kCollisionInset;
  const float fieldCenterY = worldCenter().y;
  const float fieldMinY = fieldCenterY - kForcefieldThickness * 0.5f;
  const float fieldMaxY = fieldCenterY + kForcefieldThickness * 0.5f;
  return playerMaxY > fieldMinY && playerMinY < fieldMaxY;
}

std::optional<BombsitePrediction> predictBombsiteImpact(const World& world, float orbitPhase, float orbitYaw,
                                                        float orbitSpeedScale) {
  const glm::vec3 center = worldCenter();
  glm::vec3 position = satellitePositionAtAngle(orbitPhase, orbitYaw, orbitSpeedScale);
  const glm::vec3 radial = position - center;
  const float orbitRadius = satelliteOrbitRadius(orbitSpeedScale);
  const float orbitSpeed = glm::two_pi<float>() * orbitRadius * orbitSpeedScale / kSatelliteOrbitPeriod;
  glm::vec3 velocity = satelliteTangentAtAngle(orbitPhase, orbitYaw, orbitSpeedScale) * orbitSpeed;

  const float fieldCenterY = center.y;
  const float fieldHalfWidth = static_cast<float>(kWorldX) * kBlockSize * kForcefieldOversize * 0.5f;
  const float fieldHalfDepth = static_cast<float>(kWorldZ) * kBlockSize * kForcefieldOversize * 0.5f;

  for (float elapsed = 0.0f; elapsed < kBombDropMaxTime; elapsed += kBombDropStep) {
    const glm::vec3 previous = position;
    const glm::vec3 toCenter = glm::normalize(center - position);
    velocity += toCenter * kBombDropGravity * kBombDropStep;
    position += velocity * kBombDropStep;

    if ((previous.y - fieldCenterY) * (position.y - fieldCenterY) <= 0.0f) {
      const float denom = position.y - previous.y;
      if (std::abs(denom) > 0.0001f) {
        const float t = std::clamp((fieldCenterY - previous.y) / denom, 0.0f, 1.0f);
        const glm::vec3 hitPoint = glm::mix(previous, position, t);
        if (std::abs(hitPoint.x - center.x) <= fieldHalfWidth && std::abs(hitPoint.z - center.z) <= fieldHalfDepth) {
          return BombsitePrediction{hitPoint, true};
        }
      }
    }

    const glm::vec3 segment = position - previous;
    const float distance = glm::length(segment);
    if (distance > 0.0001f) {
      if (const auto hit = raycast(world, previous, segment / distance, distance)) {
        return BombsitePrediction{blockCenterToWorld(hit->block), false};
      }
    }
  }

  return std::nullopt;
}

void tryBreakBlock(AppState& state) {
  if (!state.hoveredBlock.has_value()) {
    return;
  }

  const glm::ivec3 block = state.hoveredBlock->block;
  if (block.y <= 0) {
    return;
  }

  state.world.set(block.x, block.y, block.z, Air);
  markDirtyAroundBlock(state, block.x, block.z);
  triggerHandSwing(state);
  triggerCameraSnap(state);
  resetMining(state);
}

void tryPlaceBlock(AppState& state) {
  if (!state.hoveredBlock.has_value()) {
    return;
  }

  const glm::ivec3 place = state.hoveredBlock->previous;
  if (!state.world.inBounds(place.x, place.y, place.z) || state.world.get(place.x, place.y, place.z) != Air) {
    return;
  }
  if (playerIntersectsBlock(state.player.position, place)) {
    return;
  }

  state.world.set(place.x, place.y, place.z, state.selectedBlock);
  markDirtyAroundBlock(state, place.x, place.z);
  triggerHandSwing(state);
  triggerCameraSnap(state);
  resetPlacement(state);
}

bool missileHitsTarget(const glm::vec3& impactPos) {
  const glm::vec3 center = targetCenter();
  const glm::vec2 delta(impactPos.x - center.x, impactPos.z - center.z);
  const float halfThickness = 1.5f * kBlockSize;
  const float armLength = (std::min(kWorldX, kWorldZ) / 6.0f + 0.5f) * kBlockSize;
  const bool verticalArm = std::abs(delta.x) <= halfThickness && std::abs(delta.y) <= armLength;
  const bool horizontalArm = std::abs(delta.y) <= halfThickness && std::abs(delta.x) <= armLength;
  return verticalArm || horizontalArm;
}

void launchMissile(AppState& state) {
  if (!state.launcherEquipped || state.missile.active || state.missile.exploding) {
    return;
  }

  const MissileSolution solution = buildMissileSolution(state);

  state.missile.active = true;
  state.missile.exploding = false;
  state.missile.hitTarget = false;
  state.missile.launchPos = solution.launchPos;
  state.missile.startOrbitPos = solution.startOrbitPos;
  state.missile.endOrbitPos = solution.endOrbitPos;
  state.missile.impactPos = solution.impactPos;
  state.missile.currentPos = solution.launchPos;
  state.missile.progress = 0.0f;
  state.missile.duration = solution.duration;
  state.missile.explosionAge = 0.0f;
  state.missileAim.charging = false;
  triggerHandSwing(state);
}

void updateMissile(AppState& state, float deltaTime) {
  if (state.missile.active) {
    state.missile.progress = std::min(1.0f, state.missile.progress + deltaTime / state.missile.duration);
    const float t = state.missile.progress;
    const MissileSolution solution{
        state.missile.launchPos, state.missile.startOrbitPos, state.missile.endOrbitPos,
        state.missile.impactPos, state.missile.duration};
    state.missile.currentPos = missilePositionAt(solution, t);

    if (state.missile.progress >= 1.0f) {
      state.missile.active = false;
      state.missile.exploding = true;
      state.missile.explosionAge = 0.0f;
      state.missile.currentPos = state.missile.impactPos;
      state.missile.hitTarget = missileHitsTarget(state.missile.impactPos);
      clearExplosionCrater(state, state.missile.impactPos, kMissileExplosionRadius);
    }
  } else if (state.missile.exploding) {
    state.missile.explosionAge += deltaTime;
    if (state.missile.explosionAge >= kMissileExplosionDuration) {
      state.missile.exploding = false;
      state.missile.hitTarget = false;
    }
  }
}

void dropAtomicBomb(AppState& state) {
  if (state.atomicBomb.active || state.atomicBomb.exploding) {
    return;
  }

  const float orbitRadius = satelliteOrbitRadius(state.satelliteOrbitSpeed);
  state.atomicBomb.active = true;
  state.atomicBomb.bouncing = false;
  state.atomicBomb.exploding = false;
  state.atomicBomb.hitForcefield = false;
  state.atomicBomb.position =
      satellitePositionAtAngle(state.satelliteOrbitPhase, state.satelliteOrbitYaw, state.satelliteOrbitSpeed);
  state.atomicBomb.velocity =
      satelliteTangentAtAngle(state.satelliteOrbitPhase, state.satelliteOrbitYaw, state.satelliteOrbitSpeed) *
      (glm::two_pi<float>() * orbitRadius * state.satelliteOrbitSpeed / kSatelliteOrbitPeriod);
  state.atomicBomb.impactPos = state.atomicBomb.position;
  state.atomicBomb.bounceAge = 0.0f;
  state.atomicBomb.explosionAge = 0.0f;
  state.atomicBomb.trailCount = 0;
  pushAtomicBombTrail(state.atomicBomb, state.atomicBomb.position);
  triggerHandSwing(state);
}

void updateAtomicBomb(AppState& state, float deltaTime) {
  if (state.atomicBomb.active) {
    const glm::vec3 center = worldCenter();
    const float fieldCenterY = center.y;
    const float fieldHalfWidth = static_cast<float>(kWorldX) * kBlockSize * kForcefieldOversize * 0.5f;
    const float fieldHalfDepth = static_cast<float>(kWorldZ) * kBlockSize * kForcefieldOversize * 0.5f;

    for (float remaining = deltaTime; remaining > 0.0f && state.atomicBomb.active;) {
      const float stepTime = std::min(kBombDropStep, remaining);
      remaining -= stepTime;

      const glm::vec3 previous = state.atomicBomb.position;
      const glm::vec3 toCenter = glm::normalize(center - state.atomicBomb.position);
      state.atomicBomb.velocity += toCenter * kBombDropGravity * stepTime;
      state.atomicBomb.position += state.atomicBomb.velocity * stepTime;

      if ((previous.y - fieldCenterY) * (state.atomicBomb.position.y - fieldCenterY) <= 0.0f) {
        const float denom = state.atomicBomb.position.y - previous.y;
        if (std::abs(denom) > 0.0001f) {
          const float t = std::clamp((fieldCenterY - previous.y) / denom, 0.0f, 1.0f);
          const glm::vec3 hitPoint = glm::mix(previous, state.atomicBomb.position, t);
          if (std::abs(hitPoint.x - center.x) <= fieldHalfWidth &&
              std::abs(hitPoint.z - center.z) <= fieldHalfDepth) {
            startAtomicBombBounce(state, hitPoint, true, state.atomicBomb.velocity);
            break;
          }
        }
      }

      const glm::vec3 segment = state.atomicBomb.position - previous;
      const float distance = glm::length(segment);
      if (distance > 0.0001f) {
        if (const auto hit = raycast(state.world, previous, segment / distance, distance)) {
          startAtomicBombBounce(state, blockCenterToWorld(hit->block), false, state.atomicBomb.velocity);
          break;
        }
      }
    }

    if (state.atomicBomb.active) {
      pushAtomicBombTrail(state.atomicBomb, state.atomicBomb.position);
    }
  } else if (state.atomicBomb.bouncing) {
    state.atomicBomb.bounceAge += deltaTime;
    const float t = std::clamp(state.atomicBomb.bounceAge / kAtomicBombBounceDuration, 0.0f, 1.0f);
    const float decay = 1.0f - t;
    const float hop = std::abs(std::sin(t * glm::two_pi<float>() * 3.3f)) * decay;
    const float wobble = std::sin(t * glm::two_pi<float>() * 7.5f) * decay;
    state.atomicBomb.position =
        state.atomicBomb.impactPos +
        state.atomicBomb.bounceNormal * (hop * kAtomicBombBounceHeight) +
        state.atomicBomb.bounceDrift * (wobble * kAtomicBombBounceDrift);
    pushAtomicBombTrail(state.atomicBomb, state.atomicBomb.position);
    if (state.atomicBomb.bounceAge >= kAtomicBombBounceDuration) {
      startAtomicBombExplosion(state, state.atomicBomb.impactPos, state.atomicBomb.hitForcefield);
    }
  } else if (state.atomicBomb.exploding) {
    state.atomicBomb.explosionAge += deltaTime;
    if (state.atomicBomb.explosionAge >= kAtomicBombExplosionDuration) {
      state.atomicBomb.exploding = false;
      state.atomicBomb.hitForcefield = false;
      state.atomicBomb.trailCount = 0;
    }
  }
}

void updateHand(AppState& state, float deltaTime) {
  if (!state.hand.swinging) {
    return;
  }

  state.hand.swingTime += deltaTime;
  if (state.hand.swingTime >= kHandSwingDuration) {
    state.hand.swingTime = 0.0f;
    state.hand.swinging = false;
  }
}

void updateCameraFeedback(AppState& state, float deltaTime) {
  if (state.cameraFx.thumping) {
    state.cameraFx.thumpTime += deltaTime;
    if (state.cameraFx.thumpTime >= kCameraThumpDuration) {
      state.cameraFx.thumpTime = 0.0f;
      state.cameraFx.thumping = false;
    }
  }

  if (state.cameraFx.snapping) {
    state.cameraFx.snapTime += deltaTime;
    if (state.cameraFx.snapTime >= kCameraSnapDuration) {
      state.cameraFx.snapTime = 0.0f;
      state.cameraFx.snapping = false;
    }
  }
}

void framebufferSizeCallback(GLFWwindow*, int width, int height) {
  glViewport(0, 0, width, height);
}

void mouseCallback(GLFWwindow*, double xpos, double ypos) {
  if (!gState || !gState->input.captureMouse) {
    return;
  }

  if (gState->input.firstMouse) {
    gState->input.lastMouseX = xpos;
    gState->input.lastMouseY = ypos;
    gState->input.firstMouse = false;
  }

  const double xoffset = xpos - gState->input.lastMouseX;
  const double yoffset = gState->input.lastMouseY - ypos;
  gState->input.lastMouseX = xpos;
  gState->input.lastMouseY = ypos;

  gState->player.yaw += static_cast<float>(xoffset) * kMouseSensitivity;
  gState->player.pitch += static_cast<float>(yoffset) * kMouseSensitivity;
  gState->player.pitch = std::clamp(gState->player.pitch, -89.0f, 89.0f);
}

void toggleMouseCapture(GLFWwindow* window, InputState& input, bool captured) {
  input.captureMouse = captured;
  input.firstMouse = true;
  glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

void moveAxis(const World& world, glm::vec3& position, float delta, int axis, bool& blocked) {
  if (delta == 0.0f) {
    return;
  }

  glm::vec3 candidate = position;
  candidate[axis] += delta;
  if (!collidesAt(world, candidate)) {
    position = candidate;
  } else {
    blocked = true;
  }
}

void updateMovement(GLFWwindow* window, AppState& state, float deltaTime) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS && state.input.captureMouse) {
    toggleMouseCapture(window, state.input, false);
  }
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !state.input.captureMouse) {
    toggleMouseCapture(window, state.input, true);
  }

  const glm::vec3 front = cameraFront(state.player);
  glm::vec3 flatFront(front.x, 0.0f, front.z);
  if (glm::length(flatFront) < 0.0001f) {
    flatFront = glm::vec3(0.0f, 0.0f, -1.0f);
  } else {
    flatFront = glm::normalize(flatFront);
  }
  const glm::vec3 right = glm::normalize(glm::cross(flatFront, glm::vec3(0.0f, 1.0f, 0.0f)));

  glm::vec3 wishDir(0.0f);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) wishDir += flatFront;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) wishDir -= flatFront;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) wishDir -= right;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) wishDir += right;

  if (glm::length(wishDir) > 0.0f) {
    wishDir = glm::normalize(wishDir);
  }

  const bool sprinting = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
  const float moveSpeed = kWalkSpeed * (sprinting ? kSprintMultiplier : 1.0f);
  state.player.velocity.x = wishDir.x * moveSpeed;
  state.player.velocity.z = wishDir.z * moveSpeed;

  const bool jumpPressed = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
  if (jumpPressed && !state.input.jumpHeldLastFrame && state.player.onGround) {
    state.player.velocity.y = kJumpVelocity;
    state.player.onGround = false;
  }
  state.input.jumpHeldLastFrame = jumpPressed;

  state.player.velocity.y -= kGravity * deltaTime;
  if (state.player.velocity.y < -30.0f) {
    state.player.velocity.y = -30.0f;
  }

  glm::vec3 candidate = state.player.position;
  bool blockedX = false;
  bool blockedY = false;
  bool blockedZ = false;

  moveAxis(state.world, candidate, state.player.velocity.x * deltaTime, 0, blockedX);
  moveAxis(state.world, candidate, state.player.velocity.y * deltaTime, 1, blockedY);
  moveAxis(state.world, candidate, state.player.velocity.z * deltaTime, 2, blockedZ);

  state.player.position = candidate;

  if (blockedX) state.player.velocity.x = 0.0f;
  if (blockedZ) state.player.velocity.z = 0.0f;
  if (blockedY) {
    if (state.player.velocity.y < 0.0f) {
      state.player.onGround = true;
    }
    state.player.velocity.y = 0.0f;
  } else {
    state.player.onGround = false;
  }

  const float minX = kPlayerRadius;
  const float maxX = static_cast<float>(kWorldX) * kBlockSize - kPlayerRadius;
  const float minZ = kPlayerRadius;
  const float maxZ = static_cast<float>(kWorldZ) * kBlockSize - kPlayerRadius;
  state.player.position.x = std::clamp(state.player.position.x, minX, maxX);
  state.player.position.z = std::clamp(state.player.position.z, minZ, maxZ);

  if (state.input.captureMouse) {
    if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) {
      state.satelliteOrbitYawTarget -= kSatelliteOrbitAdjustSpeed * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
      state.satelliteOrbitYawTarget += kSatelliteOrbitAdjustSpeed * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) {
      state.satelliteOrbitSpeedTarget -= kSatelliteOrbitSpeedAdjustRate * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) {
      state.satelliteOrbitSpeedTarget += kSatelliteOrbitSpeedAdjustRate * deltaTime;
    }
    state.satelliteOrbitSpeedTarget =
        std::clamp(state.satelliteOrbitSpeedTarget, kSatelliteOrbitSpeedMin, kSatelliteOrbitSpeedMax);
    if (state.satelliteOrbitYawTarget > glm::pi<float>()) {
      state.satelliteOrbitYawTarget -= glm::two_pi<float>();
    } else if (state.satelliteOrbitYawTarget < -glm::pi<float>()) {
      state.satelliteOrbitYawTarget += glm::two_pi<float>();
    }
  }

  float yawDelta = state.satelliteOrbitYawTarget - state.satelliteOrbitYaw;
  if (yawDelta > glm::pi<float>()) {
    yawDelta -= glm::two_pi<float>();
  } else if (yawDelta < -glm::pi<float>()) {
    yawDelta += glm::two_pi<float>();
  }
  state.satelliteOrbitYaw += yawDelta * std::min(1.0f, deltaTime * kSatelliteOrbitSmoothing);
  if (state.satelliteOrbitYaw > glm::pi<float>()) {
    state.satelliteOrbitYaw -= glm::two_pi<float>();
  } else if (state.satelliteOrbitYaw < -glm::pi<float>()) {
    state.satelliteOrbitYaw += glm::two_pi<float>();
  }
  state.satelliteOrbitSpeed +=
      (state.satelliteOrbitSpeedTarget - state.satelliteOrbitSpeed) *
      std::min(1.0f, deltaTime * kSatelliteOrbitSpeedSmoothing);
  state.satelliteOrbitPhase += glm::two_pi<float>() * (state.satelliteOrbitSpeed / kSatelliteOrbitPeriod) * deltaTime;
  if (state.satelliteOrbitPhase > glm::two_pi<float>()) {
    state.satelliteOrbitPhase = std::fmod(state.satelliteOrbitPhase, glm::two_pi<float>());
  }

  if (playerTouchesForcefield(state.player.position) || state.player.position.y < 2.0f * kBlockSize) {
    state.player.position = state.spawnPosition;
    state.player.velocity = glm::vec3(0.0f);
    state.player.onGround = false;
  }
}

void updateHoveredBlock(AppState& state) {
  state.hoveredBlock = raycast(state.world, eyePosition(state.player), cameraFront(state.player), kReach);
}

void handleBlockInput(GLFWwindow* window, AppState& state, float deltaTime) {
  const bool leftPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  const bool rightPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  if (state.input.captureMouse) {
    if (leftPressed) {
      if (state.hoveredBlock.has_value() && state.hoveredBlock->block.y > 0) {
        if (!state.mining.active || !sameBlock(state.mining.block, state.hoveredBlock->block)) {
          state.mining.active = true;
          state.mining.block = state.hoveredBlock->block;
          state.mining.type = state.hoveredBlock->type;
          state.mining.progress = 0.0f;
          state.mining.swingCooldown = 0.0f;
        }

        state.mining.progress += deltaTime;
        state.mining.swingCooldown -= deltaTime;
        if (state.mining.swingCooldown <= 0.0f) {
          triggerHandSwing(state);
          triggerCameraThump(state);
          state.mining.swingCooldown = kMiningSwingInterval;
        }

        if (state.mining.progress >= miningDurationFor(state.mining.type)) {
          tryBreakBlock(state);
          updateHoveredBlock(state);
        }
      } else {
        resetMining(state);
      }
    } else {
      resetMining(state);
    }
    if (rightPressed) {
      if (state.hoveredBlock.has_value()) {
        const glm::ivec3 place = state.hoveredBlock->previous;
        const bool canPlace =
            state.world.inBounds(place.x, place.y, place.z) &&
            state.world.get(place.x, place.y, place.z) == Air &&
            !playerIntersectsBlock(state.player.position, place);
        if (canPlace) {
          if (!state.placing.active || !sameBlock(state.placing.block, place) || state.placing.type != state.selectedBlock) {
            state.placing.active = true;
            state.placing.block = place;
            state.placing.type = state.selectedBlock;
            state.placing.progress = 0.0f;
            state.placing.swingCooldown = 0.0f;
          }

          state.placing.progress += deltaTime;
          state.placing.swingCooldown -= deltaTime;
          if (state.placing.swingCooldown <= 0.0f) {
            triggerHandSwing(state);
            triggerCameraThump(state);
            state.placing.swingCooldown = kPlacementSwingInterval;
          }

          if (state.placing.progress >= placementDurationFor(state.placing.type)) {
            tryPlaceBlock(state);
            updateHoveredBlock(state);
            resetMining(state);
          }
        } else {
          resetPlacement(state);
        }
      } else {
        resetPlacement(state);
      }
    } else {
      resetPlacement(state);
    }
  } else {
    resetMining(state);
    resetPlacement(state);
  }

  state.input.leftPressedLastFrame = leftPressed;
  state.input.rightPressedLastFrame = rightPressed;

  if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) state.selectedBlock = Crust;
  if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) state.selectedBlock = DarkRock;
  if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) state.selectedBlock = Ember;
}

void handleAtomicBombInput(GLFWwindow* window, AppState& state) {
  const bool bombPressed = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
  if (state.input.captureMouse && bombPressed && !state.input.atomicBombPressedLastFrame) {
    dropAtomicBomb(state);
  }
  state.input.atomicBombPressedLastFrame = bombPressed;
}

GLuint createOutlineVao(GLuint& vbo) {
  static constexpr std::array<glm::vec3, 24> kOutlineVertices = {
      glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{kBlockSize, 0.0f, 0.0f},
      glm::vec3{kBlockSize, 0.0f, 0.0f}, glm::vec3{kBlockSize, kBlockSize, 0.0f},
      glm::vec3{kBlockSize, kBlockSize, 0.0f}, glm::vec3{0.0f, kBlockSize, 0.0f},
      glm::vec3{0.0f, kBlockSize, 0.0f}, glm::vec3{0.0f, 0.0f, 0.0f},

      glm::vec3{0.0f, 0.0f, kBlockSize}, glm::vec3{kBlockSize, 0.0f, kBlockSize},
      glm::vec3{kBlockSize, 0.0f, kBlockSize}, glm::vec3{kBlockSize, kBlockSize, kBlockSize},
      glm::vec3{kBlockSize, kBlockSize, kBlockSize}, glm::vec3{0.0f, kBlockSize, kBlockSize},
      glm::vec3{0.0f, kBlockSize, kBlockSize}, glm::vec3{0.0f, 0.0f, kBlockSize},

      glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 0.0f, kBlockSize},
      glm::vec3{kBlockSize, 0.0f, 0.0f}, glm::vec3{kBlockSize, 0.0f, kBlockSize},
      glm::vec3{kBlockSize, kBlockSize, 0.0f}, glm::vec3{kBlockSize, kBlockSize, kBlockSize},
      glm::vec3{0.0f, kBlockSize, 0.0f}, glm::vec3{0.0f, kBlockSize, kBlockSize},
  };

  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kOutlineVertices), kOutlineVertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
  glEnableVertexAttribArray(0);
  return vao;
}

GLuint createSolidCubeVao(GLuint& vbo) {
  static constexpr std::array<glm::vec3, 36> kCubeVertices = {
      glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 0.0f},
      glm::vec3{1.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f},

      glm::vec3{1.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 1.0f}, glm::vec3{0.0f, 0.0f, 1.0f},
      glm::vec3{0.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 1.0f},

      glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 0.0f},
      glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 1.0f},

      glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 1.0f},
      glm::vec3{1.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f},

      glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 0.0f},
      glm::vec3{1.0f, 1.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 1.0f},

      glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 1.0f},
      glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 0.0f, 0.0f},
  };

  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kCubeVertices), kCubeVertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
  glEnableVertexAttribArray(0);
  return vao;
}

GLuint createDynamicLineVao(GLuint& vbo, int pointCount) {
  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * static_cast<std::size_t>(pointCount), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
  glEnableVertexAttribArray(0);
  return vao;
}

void triggerHandSwing(AppState& state) {
  state.hand.swinging = true;
  state.hand.swingTime = 0.0f;
}

void triggerCameraThump(AppState& state) {
  state.cameraFx.thumping = true;
  state.cameraFx.thumpTime = 0.0f;
}

void triggerCameraSnap(AppState& state) {
  state.cameraFx.snapping = true;
  state.cameraFx.snapTime = 0.0f;
}

void resetMining(AppState& state) {
  state.mining.active = false;
  state.mining.progress = 0.0f;
  state.mining.swingCooldown = 0.0f;
  state.mining.type = Air;
}

void resetPlacement(AppState& state) {
  state.placing.active = false;
  state.placing.progress = 0.0f;
  state.placing.swingCooldown = 0.0f;
  state.placing.type = Air;
}

}  // namespace

int main() {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW.\n";
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  GLFWwindow* window = glfwCreateWindow(kWindowWidth, kWindowHeight, "MineyCraft (OpenGL)", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create window.\n";
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    std::cerr << "Failed to initialize GLAD.\n";
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }

  AppState state;
  generateWorld(state.world);
  state.spawnPosition = findSpawnPosition(state.world);
  state.player.position = state.spawnPosition;
  markAllChunksDirty(state);
  gState = &state;
  glfwSetCursorPosCallback(window, mouseCallback);
  toggleMouseCapture(window, state.input, true);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  GLuint worldProgram = 0;
  GLuint skyProgram = 0;
  GLuint outlineProgram = 0;
  GLuint colorProgram = 0;
  GLuint skyVao = 0;
  GLuint outlineVao = 0;
  GLuint outlineVbo = 0;
  GLuint solidCubeVao = 0;
  GLuint solidCubeVbo = 0;
  GLuint orbitLineVao = 0;
  GLuint orbitLineVbo = 0;
  GLuint atlasTexture = 0;

  try {
    worldProgram = createWorldProgram();
    skyProgram = createSkyProgram();
    outlineProgram = createOutlineProgram();
    colorProgram = createColorProgram();
    atlasTexture = createAtlasTexture();
    glGenVertexArrays(1, &skyVao);
    outlineVao = createOutlineVao(outlineVbo);
    solidCubeVao = createSolidCubeVao(solidCubeVbo);
    orbitLineVao = createDynamicLineVao(orbitLineVbo, kSatelliteOrbitSegments + 1);
    rebuildDirtyChunks(state);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }

  float lastFrame = static_cast<float>(glfwGetTime());
  while (!glfwWindowShouldClose(window)) {
    const float currentFrame = static_cast<float>(glfwGetTime());
    const float deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    updateMovement(window, state, deltaTime);
    updateHoveredBlock(state);
    handleBlockInput(window, state, deltaTime);
    handleAtomicBombInput(window, state);
    updateHand(state, deltaTime);
    updateCameraFeedback(state, deltaTime);
    updateAtomicBomb(state, deltaTime);
    rebuildDirtyChunks(state);

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    const float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;

    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);
    glUseProgram(skyProgram);
    glUniform1f(glGetUniformLocation(skyProgram, "uTime"), currentFrame);
    glBindVertexArray(skyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEnable(GL_DEPTH_TEST);

    glm::vec3 eye = eyePosition(state.player);
    if (state.cameraFx.thumping) {
      const float t = state.cameraFx.thumpTime / kCameraThumpDuration;
      const float pulse = std::sin(t * 3.14159265f);
      eye += glm::vec3(0.0f, -0.18f * kBlockSize * pulse, 0.0f);
    }
    if (state.cameraFx.snapping) {
      const float t = state.cameraFx.snapTime / kCameraSnapDuration;
      const float pulse = std::sin(t * 3.14159265f);
      eye += glm::vec3(0.0f, -0.28f * kBlockSize * pulse, -0.10f * kBlockSize * pulse);
    }
    const glm::mat4 projection = glm::perspective(glm::radians(72.0f), aspect, 0.1f, 240.0f);
    const glm::mat4 view = glm::lookAt(eye, eye + cameraFront(state.player), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 identity(1.0f);
    const auto drawWorld = [&](const glm::mat4& drawView,
                               const glm::mat4& drawProjection,
                               const glm::vec3& drawEye,
                               const glm::vec3& fogColor,
                               float exposure,
                               float fogDensity) {
      glUseProgram(worldProgram);
      glUniformMatrix4fv(glGetUniformLocation(worldProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniformMatrix4fv(glGetUniformLocation(worldProgram, "uView"), 1, GL_FALSE, glm::value_ptr(drawView));
      glUniformMatrix4fv(glGetUniformLocation(worldProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(drawProjection));
      glUniform3f(glGetUniformLocation(worldProgram, "uCameraPos"), drawEye.x, drawEye.y, drawEye.z);
      const glm::vec3 center = worldCenter();
      glUniform3f(glGetUniformLocation(worldProgram, "uWorldCenter"), center.x, center.y, center.z);
      glUniform3f(glGetUniformLocation(worldProgram, "uFogColor"), fogColor.x, fogColor.y, fogColor.z);
      glUniform1f(glGetUniformLocation(worldProgram, "uExposure"), exposure);
      glUniform1f(glGetUniformLocation(worldProgram, "uFogDensity"), fogDensity);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, atlasTexture);
      glUniform1i(glGetUniformLocation(worldProgram, "uAtlas"), 0);

      for (const ChunkMesh& chunk : state.chunkMeshes) {
        if (chunk.mesh.vertexCount == 0) {
          continue;
        }
        glBindVertexArray(chunk.mesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, chunk.mesh.vertexCount);
      }
    };

    const auto drawForcefield = [&](const glm::mat4& drawView,
                                    const glm::mat4& drawProjection,
                                    bool alwaysVisible) {
      const glm::vec3 center = worldCenter();
      const float fieldWidth = static_cast<float>(kWorldX) * kBlockSize * kForcefieldOversize;
      const float fieldDepth = static_cast<float>(kWorldZ) * kBlockSize * kForcefieldOversize;
      const float fieldPulse = 0.82f + std::sin(currentFrame * 1.9f) * 0.18f;
      const glm::mat4 forcefieldModel =
          glm::translate(glm::mat4(1.0f),
                         center - glm::vec3(fieldWidth * 0.5f, kForcefieldThickness * 0.5f, fieldDepth * 0.5f)) *
          glm::scale(glm::mat4(1.0f), glm::vec3(fieldWidth, kForcefieldThickness, fieldDepth));

      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(drawView));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(drawProjection));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(forcefieldModel));
      glBindVertexArray(solidCubeVao);
      glDisable(GL_CULL_FACE);
      if (alwaysVisible) {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
      }
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                  0.18f * fieldPulse,
                  1.00f * fieldPulse,
                  0.24f * fieldPulse);
      glDrawArrays(GL_TRIANGLES, 0, 36);
      if (alwaysVisible) {
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
      }
      glEnable(GL_CULL_FACE);
    };

    drawWorld(view, projection, eye, glm::vec3(0.23f, 0.05f, 0.04f), 1.0f, 0.013f);
    drawForcefield(view, projection, false);

    {
      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(view));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(projection));
      glBindVertexArray(solidCubeVao);

      if (state.atomicBomb.active || state.atomicBomb.bouncing) {
        const glm::mat4 bombModel =
            glm::translate(glm::mat4(1.0f), state.atomicBomb.position - glm::vec3(kAtomicBombSize * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kAtomicBombSize));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(bombModel));
        const float bounceFlash = state.atomicBomb.bouncing
                                      ? 0.10f * std::sin((state.atomicBomb.bounceAge / kAtomicBombBounceDuration) *
                                                         glm::two_pi<float>() * 7.5f)
                                      : 0.0f;
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    0.11f + bounceFlash, 0.14f + bounceFlash * 0.8f, 0.10f + bounceFlash * 0.4f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 coreModel =
            glm::translate(glm::mat4(1.0f), state.atomicBomb.position - glm::vec3(kAtomicBombSize * 0.22f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kAtomicBombSize * 0.44f));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(coreModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    1.0f,
                    0.78f + bounceFlash * 1.6f,
                    0.28f + bounceFlash * 0.6f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        if (state.atomicBomb.trailCount >= 2) {
          glBindVertexArray(orbitLineVao);
          glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
          glBufferSubData(GL_ARRAY_BUFFER, 0,
                          static_cast<GLsizeiptr>(sizeof(glm::vec3) * state.atomicBomb.trailCount),
                          state.atomicBomb.trail.data());
          glDisable(GL_CULL_FACE);
          glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
          glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 1.0f, 0.72f, 0.16f);
          glDrawArrays(GL_LINE_STRIP, 0, state.atomicBomb.trailCount);
          glEnable(GL_CULL_FACE);
        }
      } else if (state.atomicBomb.exploding) {
        const float t = std::clamp(state.atomicBomb.explosionAge / kAtomicBombExplosionDuration, 0.0f, 1.0f);
        const float bloom = 1.0f - std::pow(1.0f - t, 2.6f);
        const glm::vec3 flashColor =
            state.atomicBomb.hitForcefield ? glm::vec3(0.32f, 1.0f, 0.18f) : glm::vec3(1.0f, 0.92f, 0.56f);
        const glm::vec3 outerColor =
            state.atomicBomb.hitForcefield ? glm::vec3(0.08f, 0.82f, 0.18f) : glm::vec3(1.0f, 0.44f, 0.14f);
        const float coreSize = glm::mix(0.8f * kBlockSize, 0.30f * kAtomicBombBlastRadius, bloom);
        const float shellSize = glm::mix(1.1f * kBlockSize, 0.42f * kAtomicBombBlastRadius, bloom);
        const float pillarHeight = glm::mix(1.4f * kBlockSize, kAtomicBombBlastRadius * 1.35f, bloom);
        const float stemWidth = glm::mix(0.75f * kBlockSize, 1.45f * kBlockSize, bloom);
        const float capWidth = glm::mix(1.6f * kBlockSize, 0.95f * kAtomicBombBlastRadius, bloom);
        const float capHeight = glm::mix(0.9f * kBlockSize, 0.34f * kAtomicBombBlastRadius, bloom);
        const float collarWidth = glm::mix(1.0f * kBlockSize, 0.58f * kAtomicBombBlastRadius, bloom);
        const float collarHeight = glm::mix(0.45f * kBlockSize, 0.15f * kAtomicBombBlastRadius, bloom);
        const float crownLift = glm::mix(0.9f * kBlockSize, 0.82f * pillarHeight, bloom);
        const float collarLift = glm::mix(0.45f * kBlockSize, 0.56f * pillarHeight, bloom);
        const float stemLift = glm::mix(0.25f * kBlockSize, 0.42f * pillarHeight, bloom);

        glDisable(GL_CULL_FACE);

        const glm::mat4 coreModel =
            glm::translate(glm::mat4(1.0f), state.atomicBomb.impactPos - glm::vec3(coreSize * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(coreSize));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(coreModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), flashColor.r, flashColor.g, flashColor.b);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 shellModel =
            glm::translate(glm::mat4(1.0f), state.atomicBomb.impactPos - glm::vec3(shellSize * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(shellSize));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(shellModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    outerColor.r * (1.0f - t * 0.55f),
                    outerColor.g * (1.0f - t * 0.30f),
                    outerColor.b * (1.0f - t * 0.22f));
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 stemModel =
            glm::translate(glm::mat4(1.0f),
                           state.atomicBomb.impactPos -
                               glm::vec3(stemWidth * 0.5f, stemLift, stemWidth * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(stemWidth, pillarHeight, stemWidth));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(stemModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    flashColor.r * 0.86f, flashColor.g * 0.76f, flashColor.b * 0.62f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 collarModel =
            glm::translate(glm::mat4(1.0f),
                           state.atomicBomb.impactPos +
                               glm::vec3(-collarWidth * 0.5f, collarLift, -collarWidth * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(collarWidth, collarHeight, collarWidth));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(collarModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    outerColor.r * 0.74f, outerColor.g * 0.50f, outerColor.b * 0.34f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 capModel =
            glm::translate(glm::mat4(1.0f),
                           state.atomicBomb.impactPos +
                               glm::vec3(-capWidth * 0.5f, crownLift, -capWidth * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(capWidth, capHeight, capWidth));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(capModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    flashColor.r * 0.98f, flashColor.g * 0.90f, flashColor.b * 0.78f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 crownCoreModel =
            glm::translate(glm::mat4(1.0f),
                           state.atomicBomb.impactPos +
                               glm::vec3(-capWidth * 0.28f, crownLift + capHeight * 0.15f, -capWidth * 0.28f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(capWidth * 0.56f, capHeight * 0.68f, capWidth * 0.56f));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(crownCoreModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    1.0f, flashColor.g * 0.98f, flashColor.b * 0.92f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        std::array<glm::vec3, kAtomicBombRingSegments + 1> blastRing{};
        const float ringRadius = glm::mix(1.8f * kBlockSize, kAtomicBombBlastRadius, bloom);
        const float ringY = state.atomicBomb.impactPos.y + (state.atomicBomb.hitForcefield ? 0.0f : 0.35f * kBlockSize);
        for (int i = 0; i <= kAtomicBombRingSegments; ++i) {
          const float angle = (static_cast<float>(i) / static_cast<float>(kAtomicBombRingSegments)) *
                              glm::two_pi<float>();
          blastRing[static_cast<std::size_t>(i)] =
              glm::vec3(state.atomicBomb.impactPos.x + std::cos(angle) * ringRadius,
                        ringY,
                        state.atomicBomb.impactPos.z + std::sin(angle) * ringRadius);
        }
        glBindVertexArray(orbitLineVao);
        glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(blastRing), blastRing.data());
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    flashColor.r, flashColor.g * 0.95f, flashColor.b * 0.75f);
        glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(blastRing.size()));

        const std::array<glm::vec3, 8> blastSpokes = {
            state.atomicBomb.impactPos + glm::vec3(-ringRadius, 0.0f, 0.0f),
            state.atomicBomb.impactPos + glm::vec3(ringRadius, 0.0f, 0.0f),
            state.atomicBomb.impactPos + glm::vec3(0.0f, 0.0f, -ringRadius),
            state.atomicBomb.impactPos + glm::vec3(0.0f, 0.0f, ringRadius),
            state.atomicBomb.impactPos + glm::vec3(0.0f, -ringRadius * 0.35f, 0.0f),
            state.atomicBomb.impactPos + glm::vec3(0.0f, ringRadius * 0.55f, 0.0f),
            state.atomicBomb.impactPos + glm::vec3(-ringRadius * 0.62f, ringRadius * 0.25f, 0.0f),
            state.atomicBomb.impactPos + glm::vec3(ringRadius * 0.62f, ringRadius * 0.25f, 0.0f),
        };
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(blastSpokes), blastSpokes.data());
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    flashColor.r * 0.92f, flashColor.g * 0.92f, flashColor.b * 0.82f);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(blastSpokes.size()));

        glEnable(GL_CULL_FACE);
      }
    }

    {
        std::array<glm::vec3, kSatelliteOrbitSegments + 1> orbitPoints{};
        for (int i = 0; i <= kSatelliteOrbitSegments; ++i) {
          const float t = static_cast<float>(i) / static_cast<float>(kSatelliteOrbitSegments);
          const float angle = t * glm::two_pi<float>();
          orbitPoints[static_cast<std::size_t>(i)] =
              satellitePositionAtAngle(angle, state.satelliteOrbitYaw, state.satelliteOrbitSpeed);
        }

      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(view));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(projection));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
      glBindVertexArray(orbitLineVao);
      glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3) * orbitPoints.size(), orbitPoints.data());
      glDisable(GL_CULL_FACE);
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.72f, 0.26f, 0.14f);
      glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(orbitPoints.size()));
      glEnable(GL_CULL_FACE);

      const glm::vec3 satellitePos =
          satellitePositionAtAngle(state.satelliteOrbitPhase, state.satelliteOrbitYaw, state.satelliteOrbitSpeed);
      const float blinkPhase = std::sin((currentFrame / kSatelliteBlinkPeriod) * glm::two_pi<float>()) * 0.5f + 0.5f;
      const glm::mat4 satelliteModel =
          glm::translate(glm::mat4(1.0f), satellitePos - glm::vec3(kSatelliteSize * 0.5f)) *
          glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize));
      glBindVertexArray(solidCubeVao);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(satelliteModel));
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                  0.82f + blinkPhase * 0.18f,
                  0.62f + blinkPhase * 0.26f,
                  0.16f + blinkPhase * 0.18f);
      glDrawArrays(GL_TRIANGLES, 0, 36);

      const glm::vec3 beaconOffset(0.22f * kBlockSize, 0.22f * kBlockSize, 0.0f);
      const glm::mat4 beaconModel =
          glm::translate(glm::mat4(1.0f), satellitePos + beaconOffset - glm::vec3(kSatelliteBeaconSize * 0.5f)) *
          glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteBeaconSize * (0.75f + blinkPhase * 0.65f)));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(beaconModel));
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                  0.96f + blinkPhase * 0.04f,
                  0.34f + blinkPhase * 0.36f,
                  0.18f);
      glDisable(GL_CULL_FACE);
      glDrawArrays(GL_TRIANGLES, 0, 36);
      glEnable(GL_CULL_FACE);
    }

    {
      const glm::vec3 satellitePos =
          satellitePositionAtAngle(state.satelliteOrbitPhase, state.satelliteOrbitYaw, state.satelliteOrbitSpeed);
      const glm::vec3 center = worldCenter();
      const glm::vec3 forward = glm::normalize(center - satellitePos);
      glm::vec3 orbitTangent =
          satelliteTangentAtAngle(state.satelliteOrbitPhase, state.satelliteOrbitYaw, state.satelliteOrbitSpeed);
      orbitTangent = orbitTangent - forward * glm::dot(orbitTangent, forward);
      if (glm::dot(orbitTangent, orbitTangent) < 0.0001f) {
        orbitTangent = glm::vec3(0.0f, 0.0f, 1.0f) - forward * glm::dot(glm::vec3(0.0f, 0.0f, 1.0f), forward);
      }
      const glm::vec3 miniUp = glm::normalize(orbitTangent);

      const int miniSize = std::max(440, (std::min(width, height) * 2) / 5);
      const int miniWidth = miniSize;
      const int miniHeight = miniSize;
      const int miniX = width - miniWidth - 20;
      const int miniY = height - miniHeight - 20;
      const glm::mat4 miniView = glm::lookAt(satellitePos, satellitePos + forward, miniUp);
      const glm::mat4 miniProjection =
          glm::perspective(glm::radians(42.0f), static_cast<float>(miniWidth) / static_cast<float>(miniHeight), 0.1f, 260.0f);

      glEnable(GL_SCISSOR_TEST);
      glViewport(miniX, miniY, miniWidth, miniHeight);
      glScissor(miniX, miniY, miniWidth, miniHeight);
      glClearColor(0.035f, 0.01f, 0.01f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      drawWorld(miniView, miniProjection, satellitePos, glm::vec3(0.12f, 0.03f, 0.03f), 1.55f, 0.005f);
      drawForcefield(miniView, miniProjection, true);

      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(identity));
      glBindVertexArray(orbitLineVao);
      glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
      const std::array<glm::vec3, 5> miniFrame = {
          glm::vec3(-0.96f, 0.96f, 0.0f),
          glm::vec3(0.96f, 0.96f, 0.0f),
          glm::vec3(0.96f, -0.96f, 0.0f),
          glm::vec3(-0.96f, -0.96f, 0.0f),
          glm::vec3(-0.96f, 0.96f, 0.0f),
      };
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(miniFrame), miniFrame.data());
      glDisable(GL_CULL_FACE);
      glDisable(GL_DEPTH_TEST);
      glDepthMask(GL_FALSE);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.72f, 0.18f, 0.12f);
      glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(miniFrame.size()));

      if (const auto bombsite = predictBombsiteImpact(
              state.world, state.satelliteOrbitPhase, state.satelliteOrbitYaw, state.satelliteOrbitSpeed)) {
        const glm::vec4 clip = miniProjection * miniView * glm::vec4(bombsite->impactPos, 1.0f);
        if (clip.w > 0.0001f) {
          const glm::vec3 ndc = glm::vec3(clip) / clip.w;
          if (std::abs(ndc.x) <= 1.1f && std::abs(ndc.y) <= 1.1f) {
            const glm::vec3 reticleColor = bombsite->hitForcefield
                                               ? glm::vec3(0.26f, 1.0f, 0.32f)
                                               : glm::vec3(1.0f, 0.86f, 0.22f);

            const float sight = 0.060f;
            const float gap = 0.020f;
            const std::array<glm::vec3, 8> sightLines = {
                glm::vec3(ndc.x - sight, ndc.y, 0.0f), glm::vec3(ndc.x - gap, ndc.y, 0.0f),
                glm::vec3(ndc.x + gap, ndc.y, 0.0f), glm::vec3(ndc.x + sight, ndc.y, 0.0f),
                glm::vec3(ndc.x, ndc.y - sight, 0.0f), glm::vec3(ndc.x, ndc.y - gap, 0.0f),
                glm::vec3(ndc.x, ndc.y + gap, 0.0f), glm::vec3(ndc.x, ndc.y + sight, 0.0f),
            };
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(sightLines), sightLines.data());
            glUniform3f(glGetUniformLocation(colorProgram, "uColor"), reticleColor.r, reticleColor.g, reticleColor.b);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(sightLines.size()));

            constexpr int kReticleRingSegments = 24;
            std::array<glm::vec3, kReticleRingSegments + 1> ring{};
            const float ringRadius = 0.050f;
            for (int i = 0; i <= kReticleRingSegments; ++i) {
              const float a = (static_cast<float>(i) / static_cast<float>(kReticleRingSegments)) *
                              glm::two_pi<float>();
              ring[static_cast<std::size_t>(i)] =
                  glm::vec3(ndc.x + std::cos(a) * ringRadius, ndc.y + std::sin(a) * ringRadius, 0.0f);
            }
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(ring), ring.data());
            glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                        reticleColor.r * 0.88f, reticleColor.g * 0.88f, reticleColor.b * 0.88f);
            glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(ring.size()));

            std::array<glm::vec3, 6> driftTrail{};
            int driftCount = 0;
            for (int sample = 5; sample >= 0; --sample) {
              const float dt = 0.12f * static_cast<float>(sample);
              const float pastPhase =
                  state.satelliteOrbitPhase -
                  glm::two_pi<float>() * (state.satelliteOrbitSpeed / kSatelliteOrbitPeriod) * dt;
              const auto pastBombsite = predictBombsiteImpact(
                  state.world, pastPhase, state.satelliteOrbitYaw, state.satelliteOrbitSpeed);
              if (!pastBombsite.has_value()) {
                continue;
              }
              const glm::vec4 pastClip = miniProjection * miniView * glm::vec4(pastBombsite->impactPos, 1.0f);
              if (pastClip.w <= 0.0001f) {
                continue;
              }
              const glm::vec3 pastNdc = glm::vec3(pastClip) / pastClip.w;
              if (std::abs(pastNdc.x) > 1.2f || std::abs(pastNdc.y) > 1.2f) {
                continue;
              }
              driftTrail[static_cast<std::size_t>(driftCount++)] = glm::vec3(pastNdc.x, pastNdc.y, 0.0f);
            }
            if (driftCount >= 2) {
              glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(sizeof(glm::vec3) * driftCount),
                              driftTrail.data());
              glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                          reticleColor.r * 0.52f, reticleColor.g * 0.52f, reticleColor.b * 0.52f);
              glDrawArrays(GL_LINE_STRIP, 0, driftCount);
            }

            const std::array<glm::vec3, 4> centerTick = {
                glm::vec3(ndc.x - 0.010f, ndc.y, 0.0f),
                glm::vec3(ndc.x + 0.010f, ndc.y, 0.0f),
                glm::vec3(ndc.x, ndc.y - 0.010f, 0.0f),
                glm::vec3(ndc.x, ndc.y + 0.010f, 0.0f),
            };
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(centerTick), centerTick.data());
            glUniform3f(glGetUniformLocation(colorProgram, "uColor"), reticleColor.r, reticleColor.g, reticleColor.b);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(centerTick.size()));
          }
        }
      }

      glDepthMask(GL_TRUE);
      glEnable(GL_DEPTH_TEST);
      glEnable(GL_CULL_FACE);
      glViewport(0, 0, width, height);
      glDisable(GL_SCISSOR_TEST);
    }

    if (state.hoveredBlock.has_value()) {
      const glm::vec3 blockPos = blockToWorld(state.hoveredBlock->block) - glm::vec3(0.001f);
      const glm::mat4 outlineModel = glm::translate(glm::mat4(1.0f), blockPos);
      glUseProgram(outlineProgram);
      glUniformMatrix4fv(glGetUniformLocation(outlineProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(outlineModel));
      glUniformMatrix4fv(glGetUniformLocation(outlineProgram, "uView"), 1, GL_FALSE, glm::value_ptr(view));
      glUniformMatrix4fv(glGetUniformLocation(outlineProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(projection));
      glm::vec3 outlineColor(1.0f, 0.92f, 0.72f);
      if (state.mining.active && sameBlock(state.mining.block, state.hoveredBlock->block)) {
        const float progress = std::clamp(state.mining.progress / miningDurationFor(state.mining.type), 0.0f, 1.0f);
        outlineColor = glm::mix(glm::vec3(0.96f, 0.72f, 0.24f), glm::vec3(1.0f, 0.25f, 0.10f), progress);
      }
      glUniform3f(glGetUniformLocation(outlineProgram, "uColor"), outlineColor.x, outlineColor.y, outlineColor.z);

      glDisable(GL_CULL_FACE);
      glDepthMask(GL_FALSE);
      glBindVertexArray(outlineVao);
      glDrawArrays(GL_LINES, 0, 24);
      glDepthMask(GL_TRUE);
      glEnable(GL_CULL_FACE);

      const glm::ivec3 place = state.hoveredBlock->previous;
      const bool canPlace =
          state.world.inBounds(place.x, place.y, place.z) &&
          state.world.get(place.x, place.y, place.z) == Air &&
          !playerIntersectsBlock(state.player.position, place);
      if (canPlace) {
        const glm::vec3 placePos = blockToWorld(place) - glm::vec3(0.001f);
        const glm::mat4 placeModel = glm::translate(glm::mat4(1.0f), placePos);
        glUniformMatrix4fv(glGetUniformLocation(outlineProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(placeModel));

        glm::vec3 placeColor(0.58f, 0.90f, 1.00f);
        if (state.selectedBlock == Ember) {
          placeColor = glm::vec3(1.00f, 0.48f, 0.20f);
        } else if (state.selectedBlock == DarkRock) {
          placeColor = glm::vec3(0.74f, 0.72f, 0.88f);
        } else if (state.selectedBlock == Crust) {
          placeColor = glm::vec3(0.96f, 0.78f, 0.44f);
        }
        if (state.placing.active && sameBlock(state.placing.block, place) && state.placing.type == state.selectedBlock) {
          const float progress =
              std::clamp(state.placing.progress / placementDurationFor(state.placing.type), 0.0f, 1.0f);
          placeColor = glm::mix(placeColor * 0.72f, glm::vec3(1.0f, 0.96f, 0.82f), progress);
        }
        glUniform3f(glGetUniformLocation(outlineProgram, "uColor"), placeColor.x, placeColor.y, placeColor.z);
        glDepthMask(GL_FALSE);
        glBindVertexArray(outlineVao);
        glDrawArrays(GL_LINES, 0, 24);
        glDepthMask(GL_TRUE);
      }
    }

    {
      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
      glBindVertexArray(orbitLineVao);
      glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
      glDisable(GL_DEPTH_TEST);
      glDepthMask(GL_FALSE);
      glDisable(GL_CULL_FACE);

      const bool targetLocked = state.hoveredBlock.has_value();
      const glm::vec3 crosshairColor = targetLocked ? glm::vec3(1.0f, 0.92f, 0.74f) : glm::vec3(0.78f, 0.62f, 0.52f);
      const float arm = targetLocked ? 0.020f : 0.016f;
      const float gap = targetLocked ? 0.005f : 0.007f;
      const std::array<glm::vec3, 12> crosshair = {
          glm::vec3(-arm, 0.0f, 0.0f), glm::vec3(-gap, 0.0f, 0.0f),
          glm::vec3(gap, 0.0f, 0.0f), glm::vec3(arm, 0.0f, 0.0f),
          glm::vec3(0.0f, -arm, 0.0f), glm::vec3(0.0f, -gap, 0.0f),
          glm::vec3(0.0f, gap, 0.0f), glm::vec3(0.0f, arm, 0.0f),
          glm::vec3(-0.004f, -0.004f, 0.0f), glm::vec3(0.004f, -0.004f, 0.0f),
          glm::vec3(0.004f, 0.004f, 0.0f), glm::vec3(-0.004f, 0.004f, 0.0f),
      };
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(crosshair), crosshair.data());
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"), crosshairColor.x, crosshairColor.y, crosshairColor.z);
      glDrawArrays(GL_LINES, 0, 8);
      glDrawArrays(GL_LINE_LOOP, 8, 4);

      if (targetLocked) {
        const std::array<glm::vec3, 8> brackets = {
            glm::vec3(-0.030f, -0.030f, 0.0f), glm::vec3(-0.018f, -0.030f, 0.0f),
            glm::vec3(-0.030f, -0.030f, 0.0f), glm::vec3(-0.030f, -0.018f, 0.0f),
            glm::vec3(0.030f, 0.030f, 0.0f), glm::vec3(0.018f, 0.030f, 0.0f),
            glm::vec3(0.030f, 0.030f, 0.0f), glm::vec3(0.030f, 0.018f, 0.0f),
        };
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(brackets), brackets.data());
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 1.0f, 0.70f, 0.32f);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(brackets.size()));
      }

      glEnable(GL_CULL_FACE);
      glDepthMask(GL_TRUE);
      glEnable(GL_DEPTH_TEST);
    }

    {
      const float swingT = state.hand.swinging ? (state.hand.swingTime / kHandSwingDuration) : 0.0f;
      const float swingArc = std::sin(swingT * 3.14159265f);
      const float swingDrop = std::sin(swingT * 6.2831853f) * 0.08f;

      const glm::mat4 handProjection = glm::perspective(glm::radians(65.0f), aspect, 0.01f, 10.0f);
      const glm::mat4 handView(1.0f);

      glm::mat4 handModel(1.0f);
      handModel = glm::translate(handModel, glm::vec3(0.56f + swingArc * 0.06f, -0.86f - swingDrop * 0.7f, -0.92f));
      handModel = glm::rotate(handModel, glm::radians(10.0f + swingArc * 16.0f), glm::vec3(0.0f, 0.0f, 1.0f));
      handModel = glm::rotate(handModel, glm::radians(-10.0f - swingArc * 18.0f), glm::vec3(1.0f, 0.0f, 0.0f));
      handModel = glm::scale(handModel, glm::vec3(0.20f, 0.28f, 0.16f));

      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(handView));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(handProjection));
      glBindVertexArray(solidCubeVao);
      glDisable(GL_DEPTH_TEST);
      glDepthMask(GL_FALSE);
      glDisable(GL_CULL_FACE);

      glm::mat4 toolModel(1.0f);
      toolModel = glm::translate(toolModel, glm::vec3(0.78f + swingArc * 0.08f, -0.72f - swingDrop, -1.25f));
      toolModel = glm::rotate(toolModel, glm::radians(18.0f + swingArc * 22.0f), glm::vec3(0.0f, 0.0f, 1.0f));
      toolModel = glm::rotate(toolModel, glm::radians(-18.0f - swingArc * 30.0f), glm::vec3(1.0f, 0.0f, 0.0f));
      toolModel = glm::scale(toolModel, glm::vec3(0.22f, 0.9f, 0.22f));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(toolModel));
      if (state.selectedBlock == Ember) {
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.82f, 0.28f, 0.14f);
      } else if (state.selectedBlock == DarkRock) {
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.28f, 0.16f, 0.14f);
      } else {
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.56f, 0.22f, 0.12f);
      }
      glDrawArrays(GL_TRIANGLES, 0, 36);

      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(handModel));
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.86f, 0.68f, 0.56f);
      glDrawArrays(GL_TRIANGLES, 0, 36);
      glEnable(GL_CULL_FACE);
      glDepthMask(GL_TRUE);
      glEnable(GL_DEPTH_TEST);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteTextures(1, &atlasTexture);
  glDeleteBuffers(1, &orbitLineVbo);
  glDeleteVertexArrays(1, &orbitLineVao);
  glDeleteBuffers(1, &solidCubeVbo);
  glDeleteVertexArrays(1, &solidCubeVao);
  glDeleteBuffers(1, &outlineVbo);
  glDeleteVertexArrays(1, &outlineVao);
  glDeleteVertexArrays(1, &skyVao);
  for (ChunkMesh& chunk : state.chunkMeshes) {
    glDeleteBuffers(1, &chunk.mesh.vbo);
    glDeleteVertexArrays(1, &chunk.mesh.vao);
  }
  glDeleteProgram(colorProgram);
  glDeleteProgram(outlineProgram);
  glDeleteProgram(skyProgram);
  glDeleteProgram(worldProgram);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
