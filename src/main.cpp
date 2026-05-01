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
constexpr float kGamepadLookSpeed = 140.0f;
constexpr float kGamepadDeadzone = 0.18f;
constexpr float kPlayerRadius = 0.32f * kBlockSize;
constexpr float kPlayerHeight = 1.75f * kBlockSize;
constexpr float kEyeHeight = 1.62f * kBlockSize;
constexpr float kReach = 6.5f * kBlockSize;
constexpr float kStep = 0.05f * kBlockSize;
constexpr float kCollisionInset = 0.001f;
constexpr float kForcefieldThickness = 0.10f * kBlockSize;
constexpr float kForcefieldOversize = 1.155f;
constexpr float kFuelPickupAmount = 1.0f;
constexpr float kFuelDrainPerSecond = 0.05f;
constexpr float kFuelCarryMax = 12.0f;
constexpr float kFuelStartingCarry = 4.0f;
constexpr float kPlayerMaxHealth = 100.0f;
constexpr float kPlayerVoidFatalDamage = 999.0f;
constexpr float kPlayerForcefieldFatalDamage = 999.0f;
constexpr float kPlayerAtomicBombBaseDamage = 80.0f;
constexpr float kPlayerDamageFlashDecay = 1.4f;
constexpr int kFreePlayTargetKills = 3;
constexpr int kTurnBasedRounds = 10;
constexpr float kTurnRoundDuration = 60.0f;
constexpr int kPlutoniumPerPickup = 1;
constexpr int kPlutoniumPerAtomicBomb = 2;
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
constexpr int kDynamicLineCapacity = 256;
constexpr int kSatelliteNoiseSegments = 84;
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
  Fuel = 5,
  Plutonium = 6,
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

struct SatelliteState {
  float orbitYaw = 0.0f;
  float orbitYawTarget = 0.0f;
  float orbitPhase = 0.0f;
  float orbitSpeed = 1.0f;
  float orbitSpeedTarget = 1.0f;
};

enum class GameMode : std::uint8_t {
  FreePlay = 0,
  TurnBased = 1,
};

enum class TurnPhase : std::uint8_t {
  Build = 0,
  Attack = 1,
};

struct PlayerState {
  Player avatar;
  bool invertedGravity = false;
  std::optional<RaycastHit> hoveredBlock;
  BlockType selectedBlock = Crust;
  float health = kPlayerMaxHealth;
  float damageFlash = 0.0f;
  float carriedFuel = kFuelStartingCarry;
  int carriedPlutonium = 0;
  SatelliteState satellite;
  HandState hand;
  MiningState mining;
  PlacementState placing;
  CameraFeedbackState cameraFx;
  bool jumpHeldLastFrame = false;
  bool blockCycleLeftLastFrame = false;
  bool blockCycleUpLastFrame = false;
  bool blockCycleRightLastFrame = false;
};

struct AtomicBombState {
  int ownerIndex = 0;
  bool active = false;
  bool bouncing = false;
  bool exploding = false;
  bool hitForcefield = false;
  bool damageApplied = false;
  float blastScale = 1.0f;
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

struct MatchState {
  GameMode mode = GameMode::FreePlay;
  TurnPhase phase = TurnPhase::Attack;
  int roundNumber = 1;
  float roundTimeRemaining = kTurnRoundDuration;
  bool suddenDeath = false;
  bool matchOver = false;
  int winnerIndex = -1;
};

struct AppState {
  std::array<PlayerState, 2> players;
  std::array<glm::vec3, 2> spawnPositions{};
  std::array<int, 2> scores{0, 0};
  MatchState match;
  InputState input;
  World world;
  std::array<ChunkMesh, kChunkCount> chunkMeshes;
  bool launcherEquipped = false;
  MissileAimState missileAim;
  MissileState missile;
  AtomicBombState atomicBomb;
};

AppState* gState = nullptr;
GLFWcursor* gHiddenCursor = nullptr;

void triggerHandSwing(PlayerState& playerState);
void triggerCameraThump(PlayerState& playerState);
void triggerCameraSnap(PlayerState& playerState);
void resetMining(PlayerState& playerState);
void resetPlacement(PlayerState& playerState);
bool canMineAndBuild(const AppState& state);
bool canAttack(const AppState& state);
bool satellitesOnline(const AppState& state);
void respawnPlayer(AppState& state, int playerIndex, std::optional<int> killerIndex = std::nullopt);
void applyPlayerDamage(AppState& state, int playerIndex, float amount, std::optional<int> attackerIndex = std::nullopt);
void dropAtomicBomb(AppState& state, int ownerIndex);
void updateAtomicBomb(AppState& state, float deltaTime);
void resetMatch(AppState& state, GameMode mode);

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

float gravityDirection(const PlayerState& playerState) {
  return playerState.invertedGravity ? 1.0f : -1.0f;
}

glm::vec3 eyePosition(const PlayerState& playerState) {
  return playerState.avatar.position + glm::vec3(0.0f, playerState.invertedGravity ? -kEyeHeight : kEyeHeight, 0.0f);
}

glm::vec3 playerBoundsMin(const PlayerState& playerState, const glm::vec3& position) {
  if (playerState.invertedGravity) {
    return glm::vec3(position.x - kPlayerRadius + kCollisionInset,
                     position.y - kPlayerHeight + kCollisionInset,
                     position.z - kPlayerRadius + kCollisionInset);
  }
  return glm::vec3(position.x - kPlayerRadius + kCollisionInset,
                   position.y + kCollisionInset,
                   position.z - kPlayerRadius + kCollisionInset);
}

glm::vec3 playerBoundsMax(const PlayerState& playerState, const glm::vec3& position) {
  if (playerState.invertedGravity) {
    return glm::vec3(position.x + kPlayerRadius - kCollisionInset,
                     position.y - kCollisionInset,
                     position.z + kPlayerRadius - kCollisionInset);
  }
  return glm::vec3(position.x + kPlayerRadius - kCollisionInset,
                   position.y + kPlayerHeight - kCollisionInset,
                   position.z + kPlayerRadius - kCollisionInset);
}

glm::vec3 viewUpVector(const PlayerState& playerState) {
  return glm::vec3(0.0f, playerState.invertedGravity ? -1.0f : 1.0f, 0.0f);
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
    case DarkRock: return 0.70f;
    case Ember: return 0.55f;
    case Target: return 1000.0f;
    case Fuel: return 0.60f;
    case Plutonium: return 0.75f;
    case Air: return 0.0f;
  }
  return 0.5f;
}

float placementDurationFor(BlockType type) {
  switch (type) {
    case Crust: return 0.78f;
    case DarkRock: return 2.34f;
    case Ember: return 1.02f;
    case Target: return 1000.0f;
    case Fuel: return 1000.0f;
    case Plutonium: return 1000.0f;
    case Air: return 0.0f;
  }
  return 0.90f;
}

float blockShieldingValue(BlockType type) {
  switch (type) {
    case DarkRock: return 2.0f;
    case Air: return 0.0f;
    default: return 1.0f;
  }
}

float blastShieldingBetween(const World& world, const glm::vec3& start, const glm::vec3& end) {
  const glm::vec3 delta = end - start;
  const float distance = glm::length(delta);
  if (distance <= 0.0001f) {
    return 0.0f;
  }

  const glm::vec3 direction = delta / distance;
  float shielding = 0.0f;
  glm::ivec3 previousBlock(999999);
  for (float travelled = kStep; travelled < distance - kStep; travelled += kStep) {
    const glm::ivec3 block = worldToBlock(start + direction * travelled);
    if (sameBlock(block, previousBlock)) {
      continue;
    }
    previousBlock = block;
    shielding += blockShieldingValue(world.get(block.x, block.y, block.z));
  }

  return shielding;
}

MissileSolution buildMissileSolution(const AppState& state, std::optional<float> powerOverride = std::nullopt) {
  MissileSolution solution;
  const Player& player = state.players[0].avatar;
  const glm::vec3 center = worldCenter();
  const glm::vec3 outward = glm::normalize(player.position - center);
  const glm::vec3 front = cameraFront(player);
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

  const float pitchBias = std::clamp(player.pitch / 75.0f, -1.0f, 1.0f);
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
      eyePosition(player) + outward * (kMissileLaunchLift + power * 0.5f * kBlockSize) +
      launchDir * ((0.75f + power * 0.5f) * kBlockSize);

  solution.impactPos = glm::vec3(
      std::clamp(player.position.x + aimOffset.x, 0.5f * kBlockSize,
                 static_cast<float>(kWorldX) * kBlockSize - 0.5f * kBlockSize),
      0.5f * kBlockSize,
      std::clamp(player.position.z + aimOffset.z, 0.5f * kBlockSize,
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
    uniform float uContrastBoost;

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
      float warmMetalMask =
          smoothstep(0.72, 0.90, tex.r) *
          smoothstep(0.62, 0.84, tex.g) *
          (1.0 - smoothstep(0.24, 0.42, tex.b));
      float coolResourceMask =
          smoothstep(0.62, 0.84, tex.g) *
          smoothstep(0.70, 0.90, tex.b) *
          (1.0 - smoothstep(0.28, 0.46, tex.r));
      float radialDistance = length(vWorldPos - uWorldCenter);
      float verticalHeat = smoothstep(10.0, 28.0, radialDistance) * (0.08 + topBias * 0.34);
      vec3 lit = tex * (lowLight + emberGlow + verticalHeat);
      lit *= vAo;
      lit = mix(lit * 0.56, lit * 1.52, shadowBand);
      lit = pow(max(lit, vec3(0.0)), vec3(0.82));

      float lowerHalf = smoothstep(-0.75, 0.75, uWorldCenter.y - vWorldPos.y);
      vec3 upperTint = vec3(1.00, 0.82, 0.82);
      vec3 lowerTint = vec3(0.34, 0.68, 2.24);
      vec3 lowerLift = vec3(lit.r * 0.24, lit.g * 0.64, lit.b * 1.72);
      lit = mix(lit * upperTint, lowerLift * lowerTint, lowerHalf);
      lit += vec3(0.01, 0.04, 0.12) * lowerHalf;

      float crease = clamp(1.0 - vAo, 0.0, 1.0);
      float blueRim = rim * (0.55 + shadowBand * 1.35) * lowerHalf;
      lit += vec3(0.10, 0.24, 0.58) * blueRim;
      lit = mix(lit, lit * 0.52, crease * 0.42 * lowerHalf);
      lit = mix(lit, lit * 1.28, shadowBand * 0.34 * lowerHalf);
      vec3 warmMetalLit = tex * (lowLight * 1.55 + 0.32 + verticalHeat * 0.55);
      lit = mix(lit, warmMetalLit, warmMetalMask * lowerHalf);
      vec3 coolResourceLit = tex * (lowLight * 1.42 + 0.28 + verticalHeat * 0.40);
      lit = mix(lit, coolResourceLit, coolResourceMask * lowerHalf);

      float dist = distance(uCameraPos, vWorldPos);
      float groundFog = smoothstep(13.0, -6.0, vWorldPos.y) * mix(0.12, 0.06, lowerHalf);
      float fogFactor = clamp(1.0 - exp(-(dist * uFogDensity + groundFog)), 0.0, 1.0);
      vec3 lowerFogColor = vec3(0.06, 0.14, 0.44);
      vec3 fogColor = mix(uFogColor, lowerFogColor, lowerHalf);
      vec3 color = mix(lit, fogColor, fogFactor);
      color = mix(color * 0.66, color * 1.24, shadowBand);
      color *= mix(1.0, 1.10, lowerHalf);
      color = clamp((color - vec3(0.5)) * uContrastBoost + vec3(0.5), 0.0, 1.0);
      color = mix(color, sqrt(clamp(color, 0.0, 1.0)), clamp((uContrastBoost - 1.0) * 0.52, 0.0, 0.45));

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
    uniform vec3 uTopTint;
    uniform vec3 uHorizonTint;
    uniform vec3 uPitTint;

    void main() {
      float horizonBand = smoothstep(0.10, 0.62, vUv.y);
      vec3 color = mix(uPitTint, uHorizonTint, horizonBand);
      color = mix(color, uTopTint, smoothstep(0.58, 1.0, vUv.y));

      float heatWave = sin(vUv.x * 19.0 + uTime * 0.3) * 0.01 + sin(vUv.x * 7.0 - uTime * 0.15) * 0.015;
      vec3 shimmerTint = mix(vec3(0.18, 0.05, 0.01), vec3(0.08, 0.18, 0.34), clamp(uHorizonTint.b * 1.2, 0.0, 1.0));
      color += shimmerTint * smoothstep(0.15, 0.55, vUv.y + heatWave) * (1.0 - smoothstep(0.55, 0.9, vUv.y));
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
    uniform float uAlpha;

    void main() {
      FragColor = vec4(uColor, uAlpha);
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
  writeTile(2, 1, {38, 194, 210}, {12, 92, 118}, true);
  writeTile(3, 0, {232, 214, 56}, {118, 104, 14}, true);
  return pixels;
}

GLuint createAtlasTexture() {
  constexpr int tileSize = 16;
  constexpr int gridSize = 4;
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
  constexpr float columns = 4.0f;
  constexpr float rows = 2.0f;
  const float tileU = 1.0f / columns;
  const float tileV = 1.0f / rows;
  const int tileX = tileIndex % 4;
  const int tileY = tileIndex / 4;
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
    case Ember: return 4;
    case Target: return 2;
    case Fuel: return 6;
    case Plutonium: return 3;
    case Air: return 7;
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

        const float fuelNoise = hashNoise3(x + 103, y + 211, z + 157);
        const float plutoniumNoise = hashNoise3(x + 401, y + 97, z + 281);
        if (surfaceDepth <= 3 && type != Air && type != Target && fuelNoise > 0.982f) {
          type = Fuel;
        } else if (surfaceDepth <= 3 && type != Air && type != Target && plutoniumNoise > 0.989f) {
          type = Plutonium;
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
  for (PlayerState& playerState : state.players) {
    triggerCameraThump(playerState);
    triggerCameraSnap(playerState);
  }
  if (!hitForcefield) {
    clearExplosionCrater(state, impactPos, kAtomicBombCraterRadius * state.atomicBomb.blastScale);
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
  for (PlayerState& playerState : state.players) {
    triggerCameraThump(playerState);
  }
}

bool isSolid(const World& world, int x, int y, int z) {
  return world.get(x, y, z) != Air;
}

bool collidesAt(const World& world, const PlayerState& playerState, const glm::vec3& position) {
  const glm::vec3 min = playerBoundsMin(playerState, position);
  const glm::vec3 max = playerBoundsMax(playerState, position);

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

int bottomSolidYAt(const World& world, int x, int z) {
  for (int y = 0; y < kWorldY; ++y) {
    if (world.get(x, y, z) != Air) {
      return y;
    }
  }
  return 0;
}

glm::vec3 findInvertedSpawnPosition(const World& world) {
  const int centerX = kWorldX / 2;
  const int centerZ = kWorldZ / 2;
  int bestX = centerX;
  int bestZ = centerZ;
  int bestY = bottomSolidYAt(world, centerX, centerZ);

  for (int radius = 0; radius < 8; ++radius) {
    for (int dz = -radius; dz <= radius; ++dz) {
      for (int dx = -radius; dx <= radius; ++dx) {
        const int x = std::clamp(centerX + dx, 0, kWorldX - 1);
        const int z = std::clamp(centerZ + dz, 0, kWorldZ - 1);
        const int y = bottomSolidYAt(world, x, z);
        if (y < bestY) {
          bestY = y;
          bestX = x;
          bestZ = z;
        }
      }
    }
  }

  return glm::vec3((static_cast<float>(bestX) + 0.5f) * kBlockSize,
                   static_cast<float>(bestY) * kBlockSize,
                   (static_cast<float>(bestZ) + 0.5f) * kBlockSize);
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

bool playerIntersectsBlock(const PlayerState& playerState, const glm::vec3& position, const glm::ivec3& block) {
  const glm::vec3 minA = playerBoundsMin(playerState, position);
  const glm::vec3 maxA = playerBoundsMax(playerState, position);

  const glm::vec3 minB = blockToWorld(block);
  const glm::vec3 maxB = minB + glm::vec3(kBlockSize);

  return minA.x < maxB.x && maxA.x > minB.x &&
         minA.y < maxB.y && maxA.y > minB.y &&
         minA.z < maxB.z && maxA.z > minB.z;
}

bool playerTouchesForcefield(const PlayerState& playerState, const glm::vec3& position) {
  const glm::vec3 minBounds = playerBoundsMin(playerState, position);
  const glm::vec3 maxBounds = playerBoundsMax(playerState, position);
  const float playerMinY = minBounds.y;
  const float playerMaxY = maxBounds.y;
  const float fieldCenterY = worldCenter().y;
  const float fieldMinY = fieldCenterY - kForcefieldThickness * 0.5f;
  const float fieldMaxY = fieldCenterY + kForcefieldThickness * 0.5f;
  return playerMaxY > fieldMinY && playerMinY < fieldMaxY;
}

std::optional<BombsitePrediction> predictBombsiteImpact(const World& world, const SatelliteState& satellite) {
  const glm::vec3 center = worldCenter();
  glm::vec3 position = satellitePositionAtAngle(satellite.orbitPhase, satellite.orbitYaw, satellite.orbitSpeed);
  const glm::vec3 radial = position - center;
  const float orbitRadius = satelliteOrbitRadius(satellite.orbitSpeed);
  const float orbitSpeed = glm::two_pi<float>() * orbitRadius * satellite.orbitSpeed / kSatelliteOrbitPeriod;
  glm::vec3 velocity = satelliteTangentAtAngle(satellite.orbitPhase, satellite.orbitYaw, satellite.orbitSpeed) * orbitSpeed;

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

void tryBreakBlock(AppState& state, PlayerState& playerState) {
  if (!playerState.hoveredBlock.has_value()) {
    return;
  }

  const glm::ivec3 block = playerState.hoveredBlock->block;
  if (block.y <= 0) {
    return;
  }

  if (playerState.hoveredBlock->type == Fuel) {
    playerState.carriedFuel = std::min(kFuelCarryMax, playerState.carriedFuel + kFuelPickupAmount);
  } else if (playerState.hoveredBlock->type == Plutonium) {
    playerState.carriedPlutonium += kPlutoniumPerPickup;
  }

  state.world.set(block.x, block.y, block.z, Air);
  markDirtyAroundBlock(state, block.x, block.z);
  triggerHandSwing(playerState);
  triggerCameraSnap(playerState);
  resetMining(playerState);
}

void tryPlaceBlock(AppState& state, PlayerState& playerState) {
  if (!playerState.hoveredBlock.has_value()) {
    return;
  }

  const glm::ivec3 place = playerState.hoveredBlock->previous;
  if (!state.world.inBounds(place.x, place.y, place.z) || state.world.get(place.x, place.y, place.z) != Air) {
    return;
  }
  for (const PlayerState& otherPlayer : state.players) {
    if (playerIntersectsBlock(otherPlayer, otherPlayer.avatar.position, place)) {
      return;
    }
  }
  if (playerIntersectsBlock(playerState, playerState.avatar.position, place)) {
    return;
  }

  state.world.set(place.x, place.y, place.z, playerState.selectedBlock);
  markDirtyAroundBlock(state, place.x, place.z);
  triggerHandSwing(playerState);
  triggerCameraSnap(playerState);
  resetPlacement(playerState);
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
  triggerHandSwing(state.players[0]);
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

void dropAtomicBomb(AppState& state, int ownerIndex) {
  if (state.atomicBomb.active || state.atomicBomb.exploding) {
    return;
  }

  PlayerState& owner = state.players[static_cast<std::size_t>(ownerIndex)];
  const SatelliteState& satellite = owner.satellite;
  const float orbitRadius = satelliteOrbitRadius(satellite.orbitSpeed);
  const bool fullStrength = owner.carriedPlutonium >= kPlutoniumPerAtomicBomb;
  if (fullStrength) {
    owner.carriedPlutonium -= kPlutoniumPerAtomicBomb;
  }
  state.atomicBomb.ownerIndex = ownerIndex;
  state.atomicBomb.active = true;
  state.atomicBomb.bouncing = false;
  state.atomicBomb.exploding = false;
  state.atomicBomb.hitForcefield = false;
  state.atomicBomb.damageApplied = false;
  state.atomicBomb.blastScale = fullStrength ? 1.0f : 0.5f;
  state.atomicBomb.position = satellitePositionAtAngle(satellite.orbitPhase, satellite.orbitYaw, satellite.orbitSpeed);
  state.atomicBomb.velocity =
      satelliteTangentAtAngle(satellite.orbitPhase, satellite.orbitYaw, satellite.orbitSpeed) *
      (glm::two_pi<float>() * orbitRadius * satellite.orbitSpeed / kSatelliteOrbitPeriod);
  state.atomicBomb.impactPos = state.atomicBomb.position;
  state.atomicBomb.bounceAge = 0.0f;
  state.atomicBomb.explosionAge = 0.0f;
  state.atomicBomb.trailCount = 0;
  pushAtomicBombTrail(state.atomicBomb, state.atomicBomb.position);
  triggerHandSwing(owner);
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
    if (!state.atomicBomb.damageApplied) {
      const float blastRadius = kAtomicBombBlastRadius * state.atomicBomb.blastScale;
      for (int playerIndex = 0; playerIndex < static_cast<int>(state.players.size()); ++playerIndex) {
        const glm::vec3 playerCenter =
            state.players[static_cast<std::size_t>(playerIndex)].avatar.position +
            glm::vec3(0.0f, kPlayerHeight * 0.5f, 0.0f);
        const float distance = glm::distance(playerCenter, state.atomicBomb.impactPos);
        if (distance < blastRadius) {
          const float falloff = 1.0f - std::clamp(distance / blastRadius, 0.0f, 1.0f);
          const float shielding = blastShieldingBetween(state.world, state.atomicBomb.impactPos, playerCenter);
          const float shieldingMultiplier = 1.0f / (1.0f + shielding * 0.45f);
          applyPlayerDamage(state, playerIndex,
                            kPlayerAtomicBombBaseDamage * state.atomicBomb.blastScale * falloff * shieldingMultiplier,
                            state.atomicBomb.ownerIndex);
        }
      }
      state.atomicBomb.damageApplied = true;
    }
    state.atomicBomb.explosionAge += deltaTime;
    if (state.atomicBomb.explosionAge >= kAtomicBombExplosionDuration) {
      state.atomicBomb.exploding = false;
      state.atomicBomb.hitForcefield = false;
      state.atomicBomb.damageApplied = false;
      state.atomicBomb.trailCount = 0;
    }
  }
}

void updateSatelliteFuel(AppState& state, float deltaTime) {
  if (!satellitesOnline(state)) {
    return;
  }
  for (PlayerState& playerState : state.players) {
    playerState.carriedFuel = std::max(0.0f, playerState.carriedFuel - kFuelDrainPerSecond * deltaTime);
  }
}

void respawnPlayer(AppState& state, int playerIndex, std::optional<int> killerIndex) {
  int creditedWinner = -1;
  if (killerIndex.has_value() && killerIndex.value() != playerIndex &&
      killerIndex.value() >= 0 && killerIndex.value() < static_cast<int>(state.scores.size())) {
    creditedWinner = killerIndex.value();
    state.scores[static_cast<std::size_t>(creditedWinner)] += 1;
  } else if (state.match.mode == GameMode::TurnBased && state.match.suddenDeath) {
    creditedWinner = 1 - playerIndex;
    state.scores[static_cast<std::size_t>(creditedWinner)] += 1;
  }

  if (state.match.mode == GameMode::FreePlay &&
      creditedWinner >= 0 &&
      state.scores[static_cast<std::size_t>(creditedWinner)] >= kFreePlayTargetKills) {
    state.match.matchOver = true;
    state.match.winnerIndex = creditedWinner;
  } else if (state.match.mode == GameMode::TurnBased && state.match.suddenDeath) {
    state.match.matchOver = true;
    state.match.winnerIndex = creditedWinner >= 0 ? creditedWinner : (1 - playerIndex);
  }

  if (state.match.matchOver) {
    resetMining(state.players[static_cast<std::size_t>(playerIndex)]);
    resetPlacement(state.players[static_cast<std::size_t>(playerIndex)]);
    return;
  }

  PlayerState& playerState = state.players[static_cast<std::size_t>(playerIndex)];
  playerState.avatar.position = state.spawnPositions[static_cast<std::size_t>(playerIndex)];
  playerState.avatar.velocity = glm::vec3(0.0f);
  playerState.avatar.onGround = false;
  playerState.health = kPlayerMaxHealth;
  playerState.damageFlash = 1.0f;
  resetMining(playerState);
  resetPlacement(playerState);
  triggerCameraSnap(playerState);
}

void applyPlayerDamage(AppState& state, int playerIndex, float amount, std::optional<int> attackerIndex) {
  if (amount <= 0.0f) {
    return;
  }

  PlayerState& playerState = state.players[static_cast<std::size_t>(playerIndex)];
  playerState.health = std::max(0.0f, playerState.health - amount);
  playerState.damageFlash = 1.0f;
  triggerCameraThump(playerState);
  if (playerState.health <= 0.0f) {
    respawnPlayer(state, playerIndex, attackerIndex);
  }
}

void updateHand(PlayerState& playerState, float deltaTime) {
  if (!playerState.hand.swinging) {
    return;
  }

  playerState.hand.swingTime += deltaTime;
  if (playerState.hand.swingTime >= kHandSwingDuration) {
    playerState.hand.swingTime = 0.0f;
    playerState.hand.swinging = false;
  }
}

void updateCameraFeedback(PlayerState& playerState, float deltaTime) {
  if (playerState.cameraFx.thumping) {
    playerState.cameraFx.thumpTime += deltaTime;
    if (playerState.cameraFx.thumpTime >= kCameraThumpDuration) {
      playerState.cameraFx.thumpTime = 0.0f;
      playerState.cameraFx.thumping = false;
    }
  }

  if (playerState.cameraFx.snapping) {
    playerState.cameraFx.snapTime += deltaTime;
    if (playerState.cameraFx.snapTime >= kCameraSnapDuration) {
      playerState.cameraFx.snapTime = 0.0f;
      playerState.cameraFx.snapping = false;
    }
  }

  playerState.damageFlash = std::max(0.0f, playerState.damageFlash - deltaTime * kPlayerDamageFlashDecay);
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

  gState->players[0].avatar.yaw += static_cast<float>(xoffset) * kMouseSensitivity;
  gState->players[0].avatar.pitch += static_cast<float>(yoffset) * kMouseSensitivity;
  gState->players[0].avatar.pitch = std::clamp(gState->players[0].avatar.pitch, -89.0f, 89.0f);
}

void toggleMouseCapture(GLFWwindow* window, InputState& input, bool captured) {
  input.captureMouse = captured;
  input.firstMouse = true;
  glfwSetCursor(window, captured ? gHiddenCursor : nullptr);
  glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

void moveAxis(const World& world, const PlayerState& playerState, glm::vec3& position, float delta, int axis, bool& blocked) {
  if (delta == 0.0f) {
    return;
  }

  glm::vec3 candidate = position;
  candidate[axis] += delta;
  if (!collidesAt(world, playerState, candidate)) {
    position = candidate;
  } else {
    blocked = true;
  }
}

float applyDeadzone(float value) {
  if (std::abs(value) < kGamepadDeadzone) {
    return 0.0f;
  }
  const float sign = value < 0.0f ? -1.0f : 1.0f;
  return sign * ((std::abs(value) - kGamepadDeadzone) / (1.0f - kGamepadDeadzone));
}

void integratePlayerMovement(PlayerState& playerState, const World& world, const glm::vec3& wishDir,
                             bool sprinting, bool jumpPressed, float deltaTime) {
  Player& player = playerState.avatar;
  const float moveSpeed = kWalkSpeed * (sprinting ? kSprintMultiplier : 1.0f);
  player.velocity.x = wishDir.x * moveSpeed;
  player.velocity.z = wishDir.z * moveSpeed;

  if (jumpPressed && !playerState.jumpHeldLastFrame && player.onGround) {
    player.velocity.y = -gravityDirection(playerState) * kJumpVelocity;
    player.onGround = false;
  }
  playerState.jumpHeldLastFrame = jumpPressed;

  player.velocity.y += gravityDirection(playerState) * kGravity * deltaTime;
  if (!playerState.invertedGravity && player.velocity.y < -30.0f) {
    player.velocity.y = -30.0f;
  } else if (playerState.invertedGravity && player.velocity.y > 30.0f) {
    player.velocity.y = 30.0f;
  }

  glm::vec3 candidate = player.position;
  bool blockedX = false;
  bool blockedY = false;
  bool blockedZ = false;

  moveAxis(world, playerState, candidate, player.velocity.x * deltaTime, 0, blockedX);
  moveAxis(world, playerState, candidate, player.velocity.y * deltaTime, 1, blockedY);
  moveAxis(world, playerState, candidate, player.velocity.z * deltaTime, 2, blockedZ);

  player.position = candidate;

  if (blockedX) player.velocity.x = 0.0f;
  if (blockedZ) player.velocity.z = 0.0f;
  if (blockedY) {
    if ((!playerState.invertedGravity && player.velocity.y < 0.0f) ||
        (playerState.invertedGravity && player.velocity.y > 0.0f)) {
      player.onGround = true;
    }
    player.velocity.y = 0.0f;
  } else {
    player.onGround = false;
  }

  const float minX = kPlayerRadius;
  const float maxX = static_cast<float>(kWorldX) * kBlockSize - kPlayerRadius;
  const float minZ = kPlayerRadius;
  const float maxZ = static_cast<float>(kWorldZ) * kBlockSize - kPlayerRadius;
  player.position.x = std::clamp(player.position.x, minX, maxX);
  player.position.z = std::clamp(player.position.z, minZ, maxZ);
}

void updateMovement(GLFWwindow* window, AppState& state, float deltaTime) {
  static bool freePlayPressedLastFrame = false;
  static bool turnBasedPressedLastFrame = false;
  static bool resetPressedLastFrame = false;

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS && state.input.captureMouse) {
    toggleMouseCapture(window, state.input, false);
  }
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !state.input.captureMouse) {
    toggleMouseCapture(window, state.input, true);
  }

  const bool freePlayPressed = glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS;
  const bool turnBasedPressed = glfwGetKey(window, GLFW_KEY_F2) == GLFW_PRESS;
  const bool resetPressed = glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
  if (freePlayPressed && !freePlayPressedLastFrame) {
    resetMatch(state, GameMode::FreePlay);
  }
  if (turnBasedPressed && !turnBasedPressedLastFrame) {
    resetMatch(state, GameMode::TurnBased);
  }
  if (resetPressed && !resetPressedLastFrame) {
    resetMatch(state, state.match.mode);
  }
  freePlayPressedLastFrame = freePlayPressed;
  turnBasedPressedLastFrame = turnBasedPressed;
  resetPressedLastFrame = resetPressed;

  if (state.match.matchOver) {
    for (PlayerState& playerState : state.players) {
      playerState.jumpHeldLastFrame = false;
    }
    return;
  }

  PlayerState& playerOne = state.players[0];
  {
    const glm::vec3 front = cameraFront(playerOne.avatar);
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
    const bool jumpPressed = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
    integratePlayerMovement(playerOne, state.world, wishDir, sprinting, jumpPressed, deltaTime);
  }

  PlayerState& playerTwo = state.players[1];
  GLFWgamepadstate gamepadState{};
  if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1) && glfwGetGamepadState(GLFW_JOYSTICK_1, &gamepadState) == GLFW_TRUE) {
    const float moveX = applyDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_X]);
    const float moveY = applyDeadzone(-gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_Y]);
    const float lookX = applyDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_X]);
    const float lookY = applyDeadzone(-gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y]);

    playerTwo.avatar.yaw += lookX * kGamepadLookSpeed * deltaTime;
    playerTwo.avatar.pitch = std::clamp(playerTwo.avatar.pitch + lookY * kGamepadLookSpeed * deltaTime,
                                        -89.0f, 89.0f);

    const glm::vec3 front = cameraFront(playerTwo.avatar);
    glm::vec3 flatFront(front.x, 0.0f, front.z);
    if (glm::length(flatFront) < 0.0001f) {
      flatFront = glm::vec3(0.0f, 0.0f, -1.0f);
    } else {
      flatFront = glm::normalize(flatFront);
    }
    const glm::vec3 right = glm::normalize(glm::cross(flatFront, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 wishDir = flatFront * moveY + right * moveX;
    if (glm::length(wishDir) > 0.0f) {
      wishDir = glm::normalize(wishDir);
    }

    const bool sprinting = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB] == GLFW_PRESS;
    const bool jumpPressed = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
    integratePlayerMovement(playerTwo, state.world, wishDir, sprinting, jumpPressed, deltaTime);

    if (canAttack(state)) {
      if (gamepadState.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS) {
        playerTwo.satellite.orbitYawTarget -= kSatelliteOrbitAdjustSpeed * deltaTime;
      }
      if (gamepadState.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS) {
        playerTwo.satellite.orbitYawTarget += kSatelliteOrbitAdjustSpeed * deltaTime;
      }
      if (gamepadState.buttons[GLFW_GAMEPAD_BUTTON_X] == GLFW_PRESS) {
        playerTwo.satellite.orbitSpeedTarget -= kSatelliteOrbitSpeedAdjustRate * deltaTime;
      }
      if (gamepadState.buttons[GLFW_GAMEPAD_BUTTON_Y] == GLFW_PRESS) {
        playerTwo.satellite.orbitSpeedTarget += kSatelliteOrbitSpeedAdjustRate * deltaTime;
      }
    }
    playerTwo.satellite.orbitSpeedTarget =
        std::clamp(playerTwo.satellite.orbitSpeedTarget, kSatelliteOrbitSpeedMin, kSatelliteOrbitSpeedMax);
    if (playerTwo.satellite.orbitYawTarget > glm::pi<float>()) {
      playerTwo.satellite.orbitYawTarget -= glm::two_pi<float>();
    } else if (playerTwo.satellite.orbitYawTarget < -glm::pi<float>()) {
      playerTwo.satellite.orbitYawTarget += glm::two_pi<float>();
    }
  } else {
    playerTwo.jumpHeldLastFrame = false;
  }

  if (state.input.captureMouse && canAttack(state)) {
    if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) {
      state.players[0].satellite.orbitYawTarget -= kSatelliteOrbitAdjustSpeed * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
      state.players[0].satellite.orbitYawTarget += kSatelliteOrbitAdjustSpeed * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) {
      state.players[0].satellite.orbitSpeedTarget -= kSatelliteOrbitSpeedAdjustRate * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) {
      state.players[0].satellite.orbitSpeedTarget += kSatelliteOrbitSpeedAdjustRate * deltaTime;
    }
    state.players[0].satellite.orbitSpeedTarget =
        std::clamp(state.players[0].satellite.orbitSpeedTarget, kSatelliteOrbitSpeedMin, kSatelliteOrbitSpeedMax);
    if (state.players[0].satellite.orbitYawTarget > glm::pi<float>()) {
      state.players[0].satellite.orbitYawTarget -= glm::two_pi<float>();
    } else if (state.players[0].satellite.orbitYawTarget < -glm::pi<float>()) {
      state.players[0].satellite.orbitYawTarget += glm::two_pi<float>();
    }
  }

  for (PlayerState& playerState : state.players) {
    float yawDelta = playerState.satellite.orbitYawTarget - playerState.satellite.orbitYaw;
    if (yawDelta > glm::pi<float>()) {
      yawDelta -= glm::two_pi<float>();
    } else if (yawDelta < -glm::pi<float>()) {
      yawDelta += glm::two_pi<float>();
    }
    playerState.satellite.orbitYaw += yawDelta * std::min(1.0f, deltaTime * kSatelliteOrbitSmoothing);
    if (playerState.satellite.orbitYaw > glm::pi<float>()) {
      playerState.satellite.orbitYaw -= glm::two_pi<float>();
    } else if (playerState.satellite.orbitYaw < -glm::pi<float>()) {
      playerState.satellite.orbitYaw += glm::two_pi<float>();
    }
    playerState.satellite.orbitSpeed +=
        (playerState.satellite.orbitSpeedTarget - playerState.satellite.orbitSpeed) *
        std::min(1.0f, deltaTime * kSatelliteOrbitSpeedSmoothing);
    playerState.satellite.orbitPhase +=
        glm::two_pi<float>() * (playerState.satellite.orbitSpeed / kSatelliteOrbitPeriod) * deltaTime;
    if (playerState.satellite.orbitPhase > glm::two_pi<float>()) {
      playerState.satellite.orbitPhase = std::fmod(playerState.satellite.orbitPhase, glm::two_pi<float>());
    }
  }

  for (int playerIndex = 0; playerIndex < static_cast<int>(state.players.size()); ++playerIndex) {
    const glm::vec3& position = state.players[static_cast<std::size_t>(playerIndex)].avatar.position;
    const PlayerState& playerState = state.players[static_cast<std::size_t>(playerIndex)];
    if (playerTouchesForcefield(playerState, position)) {
      applyPlayerDamage(state, playerIndex, kPlayerForcefieldFatalDamage);
    } else if (!playerState.invertedGravity && position.y < 2.0f * kBlockSize) {
      applyPlayerDamage(state, playerIndex, kPlayerVoidFatalDamage);
    } else if (playerState.invertedGravity && position.y > (static_cast<float>(kWorldY) - 2.0f) * kBlockSize) {
      applyPlayerDamage(state, playerIndex, kPlayerVoidFatalDamage);
    }
  }
}

void updateHoveredBlock(AppState& state, int playerIndex) {
  PlayerState& playerState = state.players[static_cast<std::size_t>(playerIndex)];
  playerState.hoveredBlock = raycast(state.world, eyePosition(playerState), cameraFront(playerState.avatar), kReach);
}

void updatePlayerBlockInput(AppState& state, PlayerState& playerState, bool digPressed, bool placePressed,
                            float deltaTime) {
  if (digPressed) {
    if (playerState.hoveredBlock.has_value() && playerState.hoveredBlock->block.y > 0) {
      if (!playerState.mining.active || !sameBlock(playerState.mining.block, playerState.hoveredBlock->block)) {
        playerState.mining.active = true;
        playerState.mining.block = playerState.hoveredBlock->block;
        playerState.mining.type = playerState.hoveredBlock->type;
        playerState.mining.progress = 0.0f;
        playerState.mining.swingCooldown = 0.0f;
      }

      playerState.mining.progress += deltaTime;
      playerState.mining.swingCooldown -= deltaTime;
      if (playerState.mining.swingCooldown <= 0.0f) {
        triggerHandSwing(playerState);
        triggerCameraThump(playerState);
        playerState.mining.swingCooldown = kMiningSwingInterval;
      }

      if (playerState.mining.progress >= miningDurationFor(playerState.mining.type)) {
        tryBreakBlock(state, playerState);
      }
    } else {
      resetMining(playerState);
    }
  } else {
    resetMining(playerState);
  }

  if (placePressed) {
    if (playerState.hoveredBlock.has_value()) {
      const glm::ivec3 place = playerState.hoveredBlock->previous;
      bool canPlace =
          state.world.inBounds(place.x, place.y, place.z) &&
          state.world.get(place.x, place.y, place.z) == Air;
      if (canPlace) {
        for (const PlayerState& otherPlayer : state.players) {
          if (playerIntersectsBlock(otherPlayer, otherPlayer.avatar.position, place)) {
            canPlace = false;
            break;
          }
        }
      }
      if (canPlace) {
        if (!playerState.placing.active || !sameBlock(playerState.placing.block, place) ||
            playerState.placing.type != playerState.selectedBlock) {
          playerState.placing.active = true;
          playerState.placing.block = place;
          playerState.placing.type = playerState.selectedBlock;
          playerState.placing.progress = 0.0f;
          playerState.placing.swingCooldown = 0.0f;
        }

        playerState.placing.progress += deltaTime;
        playerState.placing.swingCooldown -= deltaTime;
        if (playerState.placing.swingCooldown <= 0.0f) {
          triggerHandSwing(playerState);
          triggerCameraThump(playerState);
          playerState.placing.swingCooldown = kPlacementSwingInterval;
        }

        if (playerState.placing.progress >= placementDurationFor(playerState.placing.type)) {
          tryPlaceBlock(state, playerState);
          resetMining(playerState);
        }
      } else {
        resetPlacement(playerState);
      }
    } else {
      resetPlacement(playerState);
    }
  } else {
    resetPlacement(playerState);
  }
}

void handleBlockInput(GLFWwindow* window, AppState& state, float deltaTime) {
  if (!canMineAndBuild(state)) {
    for (PlayerState& playerState : state.players) {
      resetMining(playerState);
      resetPlacement(playerState);
    }
    state.input.leftPressedLastFrame = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    state.input.rightPressedLastFrame = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    state.players[1].blockCycleLeftLastFrame = false;
    state.players[1].blockCycleUpLastFrame = false;
    state.players[1].blockCycleRightLastFrame = false;
    return;
  }

  const bool leftPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  const bool rightPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  PlayerState& playerOne = state.players[0];
  if (state.input.captureMouse) {
    updatePlayerBlockInput(state, playerOne, leftPressed, rightPressed, deltaTime);
  } else {
    resetMining(playerOne);
    resetPlacement(playerOne);
  }

  state.input.leftPressedLastFrame = leftPressed;
  state.input.rightPressedLastFrame = rightPressed;

  if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) playerOne.selectedBlock = Crust;
  if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) playerOne.selectedBlock = DarkRock;

  GLFWgamepadstate gamepadState{};
  PlayerState& playerTwo = state.players[1];
  if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1) && glfwGetGamepadState(GLFW_JOYSTICK_1, &gamepadState) == GLFW_TRUE) {
    const bool digPressed = gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > 0.45f;
    const bool placePressed = gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] > 0.45f;
    updatePlayerBlockInput(state, playerTwo, digPressed, placePressed, deltaTime);

    const bool leftSelect = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] == GLFW_PRESS;
    const bool upSelect = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] == GLFW_PRESS;
    const bool rightSelect = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS;
    if (leftSelect && !playerTwo.blockCycleLeftLastFrame) playerTwo.selectedBlock = Crust;
    if ((upSelect && !playerTwo.blockCycleUpLastFrame) ||
        (rightSelect && !playerTwo.blockCycleRightLastFrame)) {
      playerTwo.selectedBlock = DarkRock;
    }
    playerTwo.blockCycleLeftLastFrame = leftSelect;
    playerTwo.blockCycleUpLastFrame = upSelect;
    playerTwo.blockCycleRightLastFrame = rightSelect;
  } else {
    resetMining(playerTwo);
    resetPlacement(playerTwo);
    playerTwo.blockCycleLeftLastFrame = false;
    playerTwo.blockCycleUpLastFrame = false;
    playerTwo.blockCycleRightLastFrame = false;
  }
}

void handleAtomicBombInput(GLFWwindow* window, AppState& state) {
  if (!canAttack(state)) {
    state.input.atomicBombPressedLastFrame = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
    return;
  }

  const bool bombPressed = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
  if (state.input.captureMouse && bombPressed && !state.input.atomicBombPressedLastFrame) {
    dropAtomicBomb(state, 0);
  }
  state.input.atomicBombPressedLastFrame = bombPressed;

  GLFWgamepadstate gamepadState{};
  static bool playerTwoBombPressedLastFrame = false;
  const bool playerTwoBombPressed =
      glfwJoystickIsGamepad(GLFW_JOYSTICK_1) &&
      glfwGetGamepadState(GLFW_JOYSTICK_1, &gamepadState) == GLFW_TRUE &&
      gamepadState.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
  if (playerTwoBombPressed && !playerTwoBombPressedLastFrame) {
    dropAtomicBomb(state, 1);
  }
  playerTwoBombPressedLastFrame = playerTwoBombPressed;
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

void triggerHandSwing(PlayerState& playerState) {
  playerState.hand.swinging = true;
  playerState.hand.swingTime = 0.0f;
}

void triggerCameraThump(PlayerState& playerState) {
  playerState.cameraFx.thumping = true;
  playerState.cameraFx.thumpTime = 0.0f;
}

void triggerCameraSnap(PlayerState& playerState) {
  playerState.cameraFx.snapping = true;
  playerState.cameraFx.snapTime = 0.0f;
}

void resetMining(PlayerState& playerState) {
  playerState.mining.active = false;
  playerState.mining.progress = 0.0f;
  playerState.mining.swingCooldown = 0.0f;
  playerState.mining.type = Air;
}

void resetPlacement(PlayerState& playerState) {
  playerState.placing.active = false;
  playerState.placing.progress = 0.0f;
  playerState.placing.swingCooldown = 0.0f;
  playerState.placing.type = Air;
}

bool isTurnBasedBuildPhase(const AppState& state) {
  return state.match.mode == GameMode::TurnBased && !state.match.suddenDeath &&
         state.match.phase == TurnPhase::Build && !state.match.matchOver;
}

bool isTurnBasedAttackPhase(const AppState& state) {
  return state.match.mode == GameMode::TurnBased &&
         (state.match.suddenDeath || state.match.phase == TurnPhase::Attack) &&
         !state.match.matchOver;
}

bool canMineAndBuild(const AppState& state) {
  if (state.match.matchOver) {
    return false;
  }
  return state.match.mode == GameMode::FreePlay || isTurnBasedBuildPhase(state);
}

bool canAttack(const AppState& state) {
  if (state.match.matchOver) {
    return false;
  }
  return state.match.mode == GameMode::FreePlay || isTurnBasedAttackPhase(state);
}

bool satellitesOnline(const AppState& state) {
  if (state.match.matchOver) {
    return false;
  }
  return state.match.mode == GameMode::FreePlay || isTurnBasedAttackPhase(state);
}

void clearCombatState(AppState& state) {
  state.atomicBomb = AtomicBombState{};
}

void initializeSatelliteState(AppState& state) {
  state.players[0].satellite = SatelliteState{};
  state.players[1].satellite = SatelliteState{};
  state.players[1].satellite.orbitYaw = glm::half_pi<float>();
  state.players[1].satellite.orbitYawTarget = glm::half_pi<float>();
  state.players[1].satellite.orbitPhase = glm::pi<float>() * 0.75f;
}

void beginSuddenDeath(AppState& state) {
  state.match.suddenDeath = true;
  state.match.phase = TurnPhase::Attack;
  state.match.roundTimeRemaining = 0.0f;
  clearCombatState(state);
  for (PlayerState& playerState : state.players) {
    playerState.health = 1.0f;
    playerState.damageFlash = 1.0f;
    resetMining(playerState);
    resetPlacement(playerState);
  }
}

void advanceTurnRound(AppState& state) {
  clearCombatState(state);
  for (PlayerState& playerState : state.players) {
    resetMining(playerState);
    resetPlacement(playerState);
  }

  if (state.match.roundNumber >= kTurnBasedRounds) {
    if (state.scores[0] != state.scores[1]) {
      state.match.matchOver = true;
      state.match.winnerIndex = state.scores[0] > state.scores[1] ? 0 : 1;
    } else {
      beginSuddenDeath(state);
    }
    return;
  }

  state.match.roundNumber += 1;
  state.match.roundTimeRemaining = kTurnRoundDuration;
  state.match.phase = (state.match.roundNumber % 2 == 1) ? TurnPhase::Build : TurnPhase::Attack;
}

void updateMatchState(AppState& state, float deltaTime) {
  if (state.match.mode != GameMode::TurnBased || state.match.matchOver || state.match.suddenDeath) {
    return;
  }

  state.match.roundTimeRemaining = std::max(0.0f, state.match.roundTimeRemaining - deltaTime);
  if (state.match.roundTimeRemaining <= 0.0f) {
    advanceTurnRound(state);
  }
}

void resetMatch(AppState& state, GameMode mode) {
  state.scores = {0, 0};
  state.match = MatchState{};
  state.match.mode = mode;
  state.match.phase = mode == GameMode::TurnBased ? TurnPhase::Build : TurnPhase::Attack;
  state.match.roundNumber = 1;
  state.match.roundTimeRemaining = kTurnRoundDuration;
  state.match.suddenDeath = false;
  state.match.matchOver = false;
  state.match.winnerIndex = -1;
  state.launcherEquipped = false;
  state.missileAim = MissileAimState{};
  state.missile = MissileState{};
  clearCombatState(state);
  initializeSatelliteState(state);

  for (int i = 0; i < static_cast<int>(state.players.size()); ++i) {
    PlayerState& playerState = state.players[static_cast<std::size_t>(i)];
    playerState.avatar.position = state.spawnPositions[static_cast<std::size_t>(i)];
    playerState.avatar.velocity = glm::vec3(0.0f);
    playerState.avatar.onGround = false;
    playerState.health = kPlayerMaxHealth;
    playerState.damageFlash = 0.0f;
    playerState.carriedFuel = kFuelStartingCarry;
    playerState.carriedPlutonium = 0;
    playerState.selectedBlock = Crust;
    playerState.jumpHeldLastFrame = false;
    playerState.blockCycleLeftLastFrame = false;
    playerState.blockCycleUpLastFrame = false;
    playerState.blockCycleRightLastFrame = false;
    playerState.hoveredBlock.reset();
    playerState.hand = HandState{};
    playerState.cameraFx = CameraFeedbackState{};
    resetMining(playerState);
    resetPlacement(playerState);
  }

  state.players[0].avatar.yaw = -90.0f;
  state.players[0].avatar.pitch = -12.0f;
  state.players[1].avatar.yaw = 90.0f;
  state.players[1].avatar.pitch = -12.0f;
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

  GLFWwindow* window = glfwCreateWindow(
      kWindowWidth, kWindowHeight, "InterPlanetary3D by matd.space", nullptr, nullptr);
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

  {
    unsigned char hiddenPixel[4] = {0, 0, 0, 0};
    GLFWimage hiddenCursorImage{};
    hiddenCursorImage.width = 1;
    hiddenCursorImage.height = 1;
    hiddenCursorImage.pixels = hiddenPixel;
    gHiddenCursor = glfwCreateCursor(&hiddenCursorImage, 0, 0);
  }

  AppState state;
  generateWorld(state.world);
  state.spawnPositions[0] = findSpawnPosition(state.world);
  state.players[0].avatar.position = state.spawnPositions[0];
  state.players[1].invertedGravity = true;
  state.spawnPositions[1] = findInvertedSpawnPosition(state.world);
  state.players[1].avatar.position = state.spawnPositions[1];
  initializeSatelliteState(state);
  resetMatch(state, GameMode::FreePlay);
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
    orbitLineVao = createDynamicLineVao(orbitLineVbo, kDynamicLineCapacity);
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
    updateHoveredBlock(state, 0);
    updateHoveredBlock(state, 1);
    handleBlockInput(window, state, deltaTime);
    handleAtomicBombInput(window, state);
    updateHand(state.players[0], deltaTime);
    updateHand(state.players[1], deltaTime);
    updateCameraFeedback(state.players[0], deltaTime);
    updateCameraFeedback(state.players[1], deltaTime);
    updateAtomicBomb(state, deltaTime);
    updateSatelliteFuel(state, deltaTime);
    updateMatchState(state, deltaTime);
    rebuildDirtyChunks(state);

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    const glm::mat4 identity(1.0f);
    const auto drawWorld = [&](const glm::mat4& drawView,
                               const glm::mat4& drawProjection,
                               const glm::vec3& drawEye,
                               const glm::vec3& fogColor,
                               float exposure,
                               float fogDensity,
                               float contrastBoost) {
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
      glUniform1f(glGetUniformLocation(worldProgram, "uContrastBoost"), contrastBoost);

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
                                    float brightness,
                                    bool alwaysVisible) {
      const glm::vec3 center = worldCenter();
      const float fieldWidth = static_cast<float>(kWorldX) * kBlockSize * kForcefieldOversize;
      const float fieldDepth = static_cast<float>(kWorldZ) * kBlockSize * kForcefieldOversize;
      const float fieldPulse = 0.82f + std::sin(currentFrame * 1.9f) * 0.18f;
      const float fieldSnarl = 0.5f + 0.5f * std::sin(currentFrame * 7.5f);
      const glm::mat4 forcefieldCoreModel =
          glm::translate(glm::mat4(1.0f),
                         center - glm::vec3(fieldWidth * 0.5f, kForcefieldThickness * 0.5f, fieldDepth * 0.5f)) *
          glm::scale(glm::mat4(1.0f), glm::vec3(fieldWidth, kForcefieldThickness, fieldDepth));
      const glm::mat4 forcefieldHaloModel =
          glm::translate(glm::mat4(1.0f),
                         center - glm::vec3(fieldWidth * 0.515f, kForcefieldThickness * 1.4f, fieldDepth * 0.515f)) *
          glm::scale(glm::mat4(1.0f), glm::vec3(fieldWidth * 1.03f, kForcefieldThickness * 2.8f, fieldDepth * 1.03f));

      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(drawView));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(drawProjection));
      glUniform1f(glGetUniformLocation(colorProgram, "uAlpha"), 1.0f);
      glBindVertexArray(solidCubeVao);
      glDisable(GL_CULL_FACE);
      if (alwaysVisible) {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
      }

      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(forcefieldHaloModel));
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                  0.10f * fieldPulse * brightness,
                  0.72f * fieldPulse * brightness,
                  0.05f * fieldPulse * brightness);
      glDrawArrays(GL_TRIANGLES, 0, 36);

      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(forcefieldCoreModel));
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                  0.06f * fieldPulse * brightness,
                  0.30f * fieldPulse * brightness,
                  0.03f * fieldPulse * brightness);
      glDrawArrays(GL_TRIANGLES, 0, 36);

      const glm::mat4 forcefieldSkinModel =
          glm::translate(glm::mat4(1.0f),
                         center - glm::vec3(fieldWidth * 0.5f, kForcefieldThickness * 0.72f, fieldDepth * 0.5f)) *
          glm::scale(glm::mat4(1.0f), glm::vec3(fieldWidth, kForcefieldThickness * 1.44f, fieldDepth));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(forcefieldSkinModel));
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                  0.30f * fieldPulse * brightness,
                  1.40f * fieldPulse * brightness,
                  0.10f * fieldPulse * brightness);
      glDrawArrays(GL_TRIANGLES, 0, 36);

      glBindVertexArray(orbitLineVao);
      glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));

      std::array<glm::vec3, 10> edgeRings{};
      const float ringYTop = center.y + kForcefieldThickness * (0.52f + fieldSnarl * 0.08f);
      const float ringYBottom = center.y - kForcefieldThickness * (0.52f + fieldSnarl * 0.08f);
      edgeRings[0] = glm::vec3(center.x - fieldWidth * 0.5f, ringYTop, center.z - fieldDepth * 0.5f);
      edgeRings[1] = glm::vec3(center.x + fieldWidth * 0.5f, ringYTop, center.z - fieldDepth * 0.5f);
      edgeRings[2] = glm::vec3(center.x + fieldWidth * 0.5f, ringYTop, center.z + fieldDepth * 0.5f);
      edgeRings[3] = glm::vec3(center.x - fieldWidth * 0.5f, ringYTop, center.z + fieldDepth * 0.5f);
      edgeRings[4] = edgeRings[0];
      edgeRings[5] = glm::vec3(center.x - fieldWidth * 0.5f, ringYBottom, center.z - fieldDepth * 0.5f);
      edgeRings[6] = glm::vec3(center.x + fieldWidth * 0.5f, ringYBottom, center.z - fieldDepth * 0.5f);
      edgeRings[7] = glm::vec3(center.x + fieldWidth * 0.5f, ringYBottom, center.z + fieldDepth * 0.5f);
      edgeRings[8] = glm::vec3(center.x - fieldWidth * 0.5f, ringYBottom, center.z + fieldDepth * 0.5f);
      edgeRings[9] = edgeRings[5];
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(edgeRings), edgeRings.data());
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                  0.72f * fieldPulse * brightness,
                  1.34f * fieldPulse * brightness,
                  0.16f * fieldPulse * brightness);
      glDrawArrays(GL_LINE_STRIP, 0, 5);
      glDrawArrays(GL_LINE_STRIP, 5, 5);

      std::array<glm::vec3, 18> scarLines{};
      for (int i = 0; i < 9; ++i) {
        const float z = center.z - fieldDepth * 0.46f + fieldDepth * 0.92f * (static_cast<float>(i) / 8.0f);
        const float xJitter = std::sin(currentFrame * 8.0f + static_cast<float>(i) * 0.9f) * fieldWidth * 0.045f;
        scarLines[static_cast<std::size_t>(i * 2)] =
            glm::vec3(center.x - fieldWidth * 0.48f + xJitter, center.y + kForcefieldThickness * (0.15f + 0.35f * std::sin(currentFrame * 5.0f + i)), z);
        scarLines[static_cast<std::size_t>(i * 2 + 1)] =
            glm::vec3(center.x + fieldWidth * 0.48f - xJitter, center.y - kForcefieldThickness * (0.15f + 0.35f * std::cos(currentFrame * 4.0f + i)), z);
      }
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(scarLines), scarLines.data());
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                  0.84f * fieldPulse * brightness,
                  1.42f * fieldPulse * brightness,
                  0.18f * fieldPulse * brightness);
      glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(scarLines.size()));

      if (alwaysVisible) {
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
      }
      glEnable(GL_CULL_FACE);
    };

    const auto drawAtomicBomb = [&](const glm::mat4& drawView,
                                    const glm::mat4& drawProjection) {
      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(drawView));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(drawProjection));
      glUniform1f(glGetUniformLocation(colorProgram, "uAlpha"), 1.0f);
      glBindVertexArray(solidCubeVao);

      if (state.atomicBomb.active || state.atomicBomb.bouncing) {
        const bool bombFromBlueSide = state.atomicBomb.ownerIndex == 1;
        const glm::vec3 shellColor = bombFromBlueSide ? glm::vec3(0.10f, 0.16f, 0.24f) : glm::vec3(0.11f, 0.14f, 0.10f);
        const glm::vec3 coreColor = bombFromBlueSide ? glm::vec3(0.62f, 0.92f, 1.0f) : glm::vec3(1.0f, 0.78f, 0.28f);
        const glm::mat4 bombModel =
            glm::translate(glm::mat4(1.0f), state.atomicBomb.position - glm::vec3(kAtomicBombSize * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kAtomicBombSize));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(bombModel));
        const float bounceFlash = state.atomicBomb.bouncing
                                      ? 0.10f * std::sin((state.atomicBomb.bounceAge / kAtomicBombBounceDuration) *
                                                         glm::two_pi<float>() * 7.5f)
                                      : 0.0f;
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    shellColor.x + bounceFlash, shellColor.y + bounceFlash * 0.8f, shellColor.z + bounceFlash * 0.4f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 coreModel =
            glm::translate(glm::mat4(1.0f), state.atomicBomb.position - glm::vec3(kAtomicBombSize * 0.22f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kAtomicBombSize * 0.44f));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(coreModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    coreColor.x,
                    coreColor.y + bounceFlash * 1.2f,
                    coreColor.z + bounceFlash * 0.6f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        if (state.atomicBomb.trailCount >= 2) {
          glBindVertexArray(orbitLineVao);
          glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
          glBufferSubData(GL_ARRAY_BUFFER, 0,
                          static_cast<GLsizeiptr>(sizeof(glm::vec3) * state.atomicBomb.trailCount),
                          state.atomicBomb.trail.data());
          glDisable(GL_CULL_FACE);
          glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
          glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                      bombFromBlueSide ? 0.54f : 1.0f,
                      bombFromBlueSide ? 0.92f : 0.72f,
                      bombFromBlueSide ? 1.0f : 0.16f);
          glDrawArrays(GL_LINE_STRIP, 0, state.atomicBomb.trailCount);
          glEnable(GL_CULL_FACE);
        }
      } else if (state.atomicBomb.exploding) {
        const float t = std::clamp(state.atomicBomb.explosionAge / kAtomicBombExplosionDuration, 0.0f, 1.0f);
        const float bloom = 1.0f - std::pow(1.0f - t, 2.6f);
        const float blastRadius = kAtomicBombBlastRadius * state.atomicBomb.blastScale;
        const glm::vec3 flashColor =
            state.atomicBomb.hitForcefield ? glm::vec3(0.32f, 1.0f, 0.18f) : glm::vec3(1.0f, 0.92f, 0.56f);
        const glm::vec3 outerColor =
            state.atomicBomb.hitForcefield ? glm::vec3(0.08f, 0.82f, 0.18f) : glm::vec3(1.0f, 0.44f, 0.14f);
        const float coreSize = glm::mix(0.8f * kBlockSize, 0.30f * blastRadius, bloom);
        const float shellSize = glm::mix(1.1f * kBlockSize, 0.42f * blastRadius, bloom);
        const float pillarHeight = glm::mix(1.4f * kBlockSize, blastRadius * 1.35f, bloom);
        const float stemWidth = glm::mix(0.75f * kBlockSize, 1.45f * kBlockSize, bloom);
        const float capWidth = glm::mix(1.6f * kBlockSize, 0.95f * blastRadius, bloom);
        const float capHeight = glm::mix(0.9f * kBlockSize, 0.34f * blastRadius, bloom);
        const float collarWidth = glm::mix(1.0f * kBlockSize, 0.58f * blastRadius, bloom);
        const float collarHeight = glm::mix(0.45f * kBlockSize, 0.15f * blastRadius, bloom);
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
        const float ringRadius = glm::mix(1.8f * kBlockSize, blastRadius, bloom);
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
    };

    const auto drawPlayerBodies = [&](int viewerIndex, const glm::mat4& drawView, const glm::mat4& drawProjection) {
      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(drawView));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(drawProjection));
      glBindVertexArray(solidCubeVao);
      for (int playerIndex = 0; playerIndex < static_cast<int>(state.players.size()); ++playerIndex) {
        if (playerIndex == viewerIndex) {
          continue;
        }
        const Player& player = state.players[static_cast<std::size_t>(playerIndex)].avatar;
        const bool inverted = state.players[static_cast<std::size_t>(playerIndex)].invertedGravity;
        const glm::vec3 bodyColor =
            playerIndex == 0 ? glm::vec3(0.96f, 0.56f, 0.36f) : glm::vec3(0.36f, 0.76f, 1.0f);
        const glm::mat4 bodyModel =
            glm::translate(glm::mat4(1.0f),
                           player.position +
                               glm::vec3(-0.38f * kBlockSize, inverted ? -1.10f * kBlockSize : 0.0f,
                                         -0.22f * kBlockSize)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(0.76f * kBlockSize, 1.10f * kBlockSize, 0.44f * kBlockSize));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(bodyModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), bodyColor.x, bodyColor.y, bodyColor.z);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 headModel =
            glm::translate(glm::mat4(1.0f),
                           player.position +
                               glm::vec3(-0.22f * kBlockSize, inverted ? -1.56f * kBlockSize : 1.12f * kBlockSize,
                                         -0.22f * kBlockSize)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(0.44f * kBlockSize));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(headModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.86f, 0.68f, 0.56f);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::vec3 headLightColor =
            playerIndex == 0 ? glm::vec3(1.00f, 0.48f, 0.18f) : glm::vec3(0.34f, 0.82f, 1.00f);
        const glm::vec3 headLightGlow =
            playerIndex == 0 ? glm::vec3(1.00f, 0.78f, 0.36f) : glm::vec3(0.62f, 0.92f, 1.00f);
        const float headLightDirection = inverted ? -1.0f : 1.0f;
        const glm::vec3 headLightCenter =
            player.position +
            glm::vec3(0.0f, headLightDirection * 1.46f * kBlockSize, 0.0f);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const float haloScale = 4.2f * kBlockSize;
        const glm::mat4 haloModel =
            glm::translate(glm::mat4(1.0f), headLightCenter - glm::vec3(haloScale * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(haloScale));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(haloModel));
        glUniform1f(glGetUniformLocation(colorProgram, "uAlpha"), 0.5f);
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), headLightGlow.x, headLightGlow.y, headLightGlow.z);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const float coreScale = 2.2f * kBlockSize;
        const glm::mat4 coreModel =
            glm::translate(glm::mat4(1.0f), headLightCenter - glm::vec3(coreScale * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(coreScale));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(coreModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), headLightColor.x, headLightColor.y, headLightColor.z);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glUniform1f(glGetUniformLocation(colorProgram, "uAlpha"), 1.0f);
        glDisable(GL_BLEND);
      }
    };

    const auto renderPlayerViewport = [&](int playerIndex, int viewportX, int viewportY, int viewportWidth,
                                          int viewportHeight) {
      PlayerState& playerState = state.players[static_cast<std::size_t>(playerIndex)];
      const float aspect = viewportHeight > 0 ? static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight) : 1.0f;

      glEnable(GL_SCISSOR_TEST);
      glViewport(viewportX, viewportY, viewportWidth, viewportHeight);
      glScissor(viewportX, viewportY, viewportWidth, viewportHeight);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glDisable(GL_DEPTH_TEST);
      glUseProgram(skyProgram);
      glUniform1f(glGetUniformLocation(skyProgram, "uTime"), currentFrame);
      if (playerState.invertedGravity) {
        glUniform3f(glGetUniformLocation(skyProgram, "uTopTint"), 0.00f, 0.02f, 0.08f);
        glUniform3f(glGetUniformLocation(skyProgram, "uHorizonTint"), 0.10f, 0.24f, 0.56f);
        glUniform3f(glGetUniformLocation(skyProgram, "uPitTint"), 0.00f, 0.00f, 0.02f);
      } else {
        glUniform3f(glGetUniformLocation(skyProgram, "uTopTint"), 0.18f, 0.03f, 0.02f);
        glUniform3f(glGetUniformLocation(skyProgram, "uHorizonTint"), 0.58f, 0.16f, 0.08f);
        glUniform3f(glGetUniformLocation(skyProgram, "uPitTint"), 0.03f, 0.00f, 0.00f);
      }
      glBindVertexArray(skyVao);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glEnable(GL_DEPTH_TEST);

      glm::vec3 eye = eyePosition(playerState);
      if (playerState.cameraFx.thumping) {
        const float t = playerState.cameraFx.thumpTime / kCameraThumpDuration;
        const float pulse = std::sin(t * 3.14159265f);
        eye += glm::vec3(0.0f, -0.18f * kBlockSize * pulse, 0.0f);
      }
      if (playerState.cameraFx.snapping) {
        const float t = playerState.cameraFx.snapTime / kCameraSnapDuration;
        const float pulse = std::sin(t * 3.14159265f);
        eye += glm::vec3(0.0f, -0.28f * kBlockSize * pulse, -0.10f * kBlockSize * pulse);
      }
      const glm::mat4 projection = glm::perspective(glm::radians(72.0f), aspect, 0.1f, 240.0f);
      const glm::mat4 view =
          glm::lookAt(eye, eye + cameraFront(playerState.avatar), viewUpVector(playerState));

      const glm::vec3 fogColor =
          playerState.invertedGravity ? glm::vec3(0.02f, 0.06f, 0.18f) : glm::vec3(0.23f, 0.05f, 0.04f);
      const float exposure = playerState.invertedGravity ? 2.15f : 1.0f;
      const float fogDensity = playerState.invertedGravity ? 0.0046f : 0.013f;

      const float contrastBoost = playerState.invertedGravity ? 1.08f : 1.0f;
      drawWorld(view, projection, eye, fogColor, exposure, fogDensity, contrastBoost);
      drawForcefield(view, projection, 1.0f, false);
      drawAtomicBomb(view, projection);
      drawPlayerBodies(playerIndex, view, projection);
      const float blinkPhase = std::sin((currentFrame / kSatelliteBlinkPeriod) * glm::two_pi<float>()) * 0.5f + 0.5f;
      const float satelliteVisualBrightness = satellitesOnline(state) ? 1.0f : 0.10f;

      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(view));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(projection));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniform1f(glGetUniformLocation(colorProgram, "uAlpha"), 1.0f);
      glBindVertexArray(orbitLineVao);
      glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
      glDisable(GL_CULL_FACE);
      for (int satelliteIndex = 0; satelliteIndex < static_cast<int>(state.players.size()); ++satelliteIndex) {
        glBindVertexArray(orbitLineVao);
        glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
        const PlayerState& satelliteOwner = state.players[static_cast<std::size_t>(satelliteIndex)];
        const SatelliteState& satellite = satelliteOwner.satellite;
        const glm::vec3 pathColor =
            satelliteIndex == 0 ? glm::vec3(0.92f, 0.28f, 0.10f) : glm::vec3(0.16f, 0.50f, 1.0f);
        std::array<glm::vec3, kSatelliteOrbitSegments + 1> orbitPoints{};
        for (int i = 0; i <= kSatelliteOrbitSegments; ++i) {
          const float t = static_cast<float>(i) / static_cast<float>(kSatelliteOrbitSegments);
          const float angle = t * glm::two_pi<float>();
          orbitPoints[static_cast<std::size_t>(i)] =
              satellitePositionAtAngle(angle, satellite.orbitYaw, satellite.orbitSpeed);
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3) * orbitPoints.size(), orbitPoints.data());
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    pathColor.x * 0.55f * satelliteVisualBrightness,
                    pathColor.y * 0.55f * satelliteVisualBrightness,
                    pathColor.z * 0.55f * satelliteVisualBrightness);
        glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(orbitPoints.size()));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    std::min(1.0f, pathColor.x * 1.15f) * satelliteVisualBrightness,
                    std::min(1.0f, pathColor.y * 1.15f) * satelliteVisualBrightness,
                    std::min(1.0f, pathColor.z * 1.15f) * satelliteVisualBrightness);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(orbitPoints.size()));

        const glm::vec3 satellitePos =
            satellitePositionAtAngle(satellite.orbitPhase, satellite.orbitYaw, satellite.orbitSpeed);
        const glm::vec3 bodyColor =
            satelliteIndex == 0 ? glm::vec3(0.84f, 0.34f, 0.10f) : glm::vec3(0.18f, 0.56f, 0.96f);
        const glm::vec3 beaconColor =
            satelliteIndex == 0 ? glm::vec3(1.0f, 0.64f, 0.20f) : glm::vec3(0.66f, 0.90f, 1.0f);
        const glm::vec3 glowColor =
            satelliteIndex == 0 ? glm::vec3(1.0f, 0.44f, 0.14f) : glm::vec3(0.42f, 0.78f, 1.0f);
        const glm::vec3 orbitNormal = glm::normalize(satellitePos - worldCenter());
        glm::vec3 orbitTangentLive = satelliteTangentAtAngle(satellite.orbitPhase, satellite.orbitYaw, satellite.orbitSpeed);
        orbitTangentLive = glm::normalize(orbitTangentLive - orbitNormal * glm::dot(orbitTangentLive, orbitNormal));
        glm::vec3 orbitBinormal = glm::cross(orbitNormal, orbitTangentLive);
        if (glm::dot(orbitBinormal, orbitBinormal) < 0.0001f) {
          orbitBinormal = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
          orbitBinormal = glm::normalize(orbitBinormal);
        }
        const glm::mat4 satelliteModel =
            glm::translate(glm::mat4(1.0f), satellitePos - glm::vec3(kSatelliteSize * 0.62f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize * 1.24f));
        glBindVertexArray(solidCubeVao);

        const glm::mat4 haloModel =
            glm::translate(glm::mat4(1.0f), satellitePos - glm::vec3(kSatelliteSize * 1.20f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize * (2.40f + blinkPhase * 0.34f)));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(haloModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    glowColor.x * 0.54f * satelliteVisualBrightness,
                    glowColor.y * 0.54f * satelliteVisualBrightness,
                    glowColor.z * 0.54f * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 haloCoreModel =
            glm::translate(glm::mat4(1.0f), satellitePos - glm::vec3(kSatelliteSize * 0.88f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize * (1.76f + blinkPhase * 0.28f)));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(haloCoreModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    glowColor.x * 0.78f * satelliteVisualBrightness,
                    glowColor.y * 0.78f * satelliteVisualBrightness,
                    glowColor.z * 0.78f * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(satelliteModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    (bodyColor.x + blinkPhase * 0.18f) * satelliteVisualBrightness,
                    (bodyColor.y + blinkPhase * 0.16f) * satelliteVisualBrightness,
                    bodyColor.z * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 coreModel =
            glm::translate(glm::mat4(1.0f), satellitePos - glm::vec3(kSatelliteSize * 0.34f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize * 0.68f));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(coreModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    std::min(1.0f, glowColor.x + 0.22f) * satelliteVisualBrightness,
                    std::min(1.0f, glowColor.y + 0.16f) * satelliteVisualBrightness,
                    std::min(1.0f, glowColor.z + 0.10f) * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 leftWingModel =
            glm::translate(glm::mat4(1.0f), satellitePos - orbitBinormal * (kSatelliteSize * 1.55f) - orbitNormal * (kSatelliteSize * 0.18f) - glm::vec3(kSatelliteSize * 0.72f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize * 1.44f, kSatelliteSize * 0.30f, kSatelliteSize * 1.44f));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(leftWingModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    bodyColor.x * 0.75f * satelliteVisualBrightness,
                    bodyColor.y * 0.75f * satelliteVisualBrightness,
                    bodyColor.z * 0.82f * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 rightWingModel =
            glm::translate(glm::mat4(1.0f), satellitePos + orbitBinormal * (kSatelliteSize * 1.55f) - orbitNormal * (kSatelliteSize * 0.18f) - glm::vec3(kSatelliteSize * 0.72f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize * 1.44f, kSatelliteSize * 0.30f, kSatelliteSize * 1.44f));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(rightWingModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    bodyColor.x * 0.75f * satelliteVisualBrightness,
                    bodyColor.y * 0.75f * satelliteVisualBrightness,
                    bodyColor.z * 0.82f * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 prowModel =
            glm::translate(glm::mat4(1.0f), satellitePos + orbitTangentLive * (kSatelliteSize * 1.48f) - glm::vec3(kSatelliteSize * 0.40f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize * 0.80f));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(prowModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    std::min(1.0f, beaconColor.x) * satelliteVisualBrightness,
                    std::min(1.0f, beaconColor.y) * satelliteVisualBrightness,
                    std::min(1.0f, beaconColor.z) * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::mat4 aftModel =
            glm::translate(glm::mat4(1.0f), satellitePos - orbitTangentLive * (kSatelliteSize * 1.42f) - glm::vec3(kSatelliteSize * 0.32f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteSize * 0.64f));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(aftModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    glowColor.x * 0.92f * satelliteVisualBrightness,
                    glowColor.y * 0.92f * satelliteVisualBrightness,
                    glowColor.z * 0.92f * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        const glm::vec3 beaconOffset(0.22f * kBlockSize, 0.22f * kBlockSize, 0.0f);
        const glm::mat4 beaconModel =
            glm::translate(glm::mat4(1.0f), satellitePos + beaconOffset - glm::vec3(kSatelliteBeaconSize * 0.5f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(kSatelliteBeaconSize * (0.75f + blinkPhase * 0.65f)));
        glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(beaconModel));
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                    beaconColor.x * satelliteVisualBrightness,
                    beaconColor.y * satelliteVisualBrightness,
                    beaconColor.z * satelliteVisualBrightness);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        if (satellitesOnline(state)) {
          const int trailPips = 20;
          for (int pip = 0; pip < trailPips; ++pip) {
            const float behind = static_cast<float>(pip) / static_cast<float>(trailPips - 1);
            float pipAngle = satellite.orbitPhase - behind * 0.58f * glm::two_pi<float>();
            if (pipAngle < 0.0f) {
              pipAngle += glm::two_pi<float>();
            }
            const glm::vec3 pipPos = satellitePositionAtAngle(pipAngle, satellite.orbitYaw, satellite.orbitSpeed);
            const glm::vec3 pipNormal = glm::normalize(pipPos - worldCenter());
            glm::vec3 pipTangent = satelliteTangentAtAngle(pipAngle, satellite.orbitYaw, satellite.orbitSpeed);
            pipTangent = glm::normalize(pipTangent - pipNormal * glm::dot(pipTangent, pipNormal));
            const float pipScale = kSatelliteBeaconSize * (1.35f - behind * 0.72f) * (1.0f + blinkPhase * 0.24f);
            const glm::mat4 pipModel =
                glm::translate(glm::mat4(1.0f), pipPos - pipTangent * (behind * 0.16f * kBlockSize) - glm::vec3(pipScale * 0.5f)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(pipScale));
            glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(pipModel));
            glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                        glowColor.x * (1.0f - behind * 0.42f),
                        glowColor.y * (1.0f - behind * 0.42f),
                        glowColor.z * (1.0f - behind * 0.42f));
            glDrawArrays(GL_TRIANGLES, 0, 36);

            const glm::vec3 sideGlowOffset = orbitBinormal * (std::sin((behind + blinkPhase) * glm::two_pi<float>() * 2.0f) * 0.22f * kBlockSize);
            const glm::mat4 sideGlowModel =
                glm::translate(glm::mat4(1.0f), pipPos + sideGlowOffset - glm::vec3(pipScale * 0.34f)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(pipScale * 0.68f));
            glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(sideGlowModel));
            glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                        glowColor.x * (0.78f - behind * 0.38f),
                        glowColor.y * (0.78f - behind * 0.38f),
                        glowColor.z * (0.78f - behind * 0.38f));
            glDrawArrays(GL_TRIANGLES, 0, 36);
          }
        }
      }
      glEnable(GL_CULL_FACE);

      const SatelliteState& playerSatellite = playerState.satellite;
      const glm::vec3 satellitePos =
          satellitePositionAtAngle(playerSatellite.orbitPhase, playerSatellite.orbitYaw, playerSatellite.orbitSpeed);

      const glm::vec3 center = worldCenter();
      const glm::vec3 satelliteForward = glm::normalize(center - satellitePos);
      glm::vec3 orbitTangent =
          satelliteTangentAtAngle(playerSatellite.orbitPhase, playerSatellite.orbitYaw, playerSatellite.orbitSpeed);
      orbitTangent = orbitTangent - satelliteForward * glm::dot(orbitTangent, satelliteForward);
      if (glm::dot(orbitTangent, orbitTangent) < 0.0001f) {
        orbitTangent = glm::vec3(0.0f, 0.0f, 1.0f) -
                       satelliteForward * glm::dot(glm::vec3(0.0f, 0.0f, 1.0f), satelliteForward);
      }
      const glm::vec3 miniUp = glm::normalize(orbitTangent);

      const int miniSize = static_cast<int>(std::max(180.0f, (std::min(viewportWidth, viewportHeight) / 3.0f) * 1.25f));
      const int miniX = viewportX + viewportWidth - miniSize - 18;
      const int miniY = viewportY + viewportHeight - miniSize - 18;
      const glm::mat4 miniView = glm::lookAt(satellitePos, satellitePos + satelliteForward, miniUp);
      const glm::mat4 miniProjection =
          glm::perspective(glm::radians(42.0f), 1.0f, 0.1f, 260.0f);

      glViewport(miniX, miniY, miniSize, miniSize);
      glScissor(miniX, miniY, miniSize, miniSize);
      glClearColor(0.035f, 0.01f, 0.01f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      const bool satelliteFueled = satellitesOnline(state) && playerState.carriedFuel > 0.001f;
      if (satelliteFueled) {
        drawWorld(miniView, miniProjection, satellitePos, glm::vec3(0.12f, 0.03f, 0.03f), 1.55f, 0.005f, 1.0f);
        drawForcefield(miniView, miniProjection, 0.72f, false);
        drawAtomicBomb(miniView, miniProjection);
        drawPlayerBodies(playerIndex, miniView, miniProjection);
      }

      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniform1f(glGetUniformLocation(colorProgram, "uAlpha"), 1.0f);
      glBindVertexArray(orbitLineVao);
      glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
      glDisable(GL_CULL_FACE);
      glDisable(GL_DEPTH_TEST);
      glDepthMask(GL_FALSE);
      const std::array<glm::vec3, 5> miniFrame = {
          glm::vec3(-0.96f, 0.96f, 0.0f), glm::vec3(0.96f, 0.96f, 0.0f),
          glm::vec3(0.96f, -0.96f, 0.0f), glm::vec3(-0.96f, -0.96f, 0.0f),
          glm::vec3(-0.96f, 0.96f, 0.0f),
      };
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(miniFrame), miniFrame.data());
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.72f, 0.18f, 0.12f);
      glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(miniFrame.size()));

      std::vector<glm::vec3> fuelTextLines;
      const auto addLine = [&](glm::vec2 a, glm::vec2 b) {
        fuelTextLines.push_back(glm::vec3(a, 0.0f));
        fuelTextLines.push_back(glm::vec3(b, 0.0f));
      };
      const auto addDigit = [&](int digit, float x, float y, float w, float h) {
        const glm::vec2 topL(x, y), topR(x + w, y), midL(x, y - h * 0.5f), midR(x + w, y - h * 0.5f),
            botL(x, y - h), botR(x + w, y - h);
        const bool seg[10][7] = {
            {true, true, true, false, true, true, true}, {false, false, true, false, false, true, false},
            {true, false, true, true, true, false, true}, {true, false, true, true, false, true, true},
            {false, true, true, true, false, true, false}, {true, true, false, true, false, true, true},
            {true, true, false, true, true, true, true}, {true, false, true, false, false, true, false},
            {true, true, true, true, true, true, true}, {true, true, true, true, false, true, true},
        };
        if (seg[digit][0]) addLine(topL, topR);
        if (seg[digit][1]) addLine(topL, midL);
        if (seg[digit][2]) addLine(topR, midR);
        if (seg[digit][3]) addLine(midL, midR);
        if (seg[digit][4]) addLine(midL, botL);
        if (seg[digit][5]) addLine(midR, botR);
        if (seg[digit][6]) addLine(botL, botR);
      };
      const auto addSlash = [&](float x, float y, float w, float h) {
        addLine(glm::vec2(x, y - h), glm::vec2(x + w, y));
      };
      const auto addNumber = [&](int value, float& cursor, float y, float w, float h, float spacing) {
        const std::string text = std::to_string(value);
        for (char ch : text) {
          addDigit(ch - '0', cursor, y, w, h);
          cursor += w + spacing;
        }
      };

      float cursor = -0.80f;
      const float textY = 0.84f;
      const float digitW = 0.11f;
      const float digitH = 0.16f;
      const float spacing = 0.035f;
      addNumber(static_cast<int>(std::ceil(playerState.carriedFuel)), cursor, textY, digitW, digitH, spacing);
      addSlash(cursor + 0.01f, textY, 0.07f, digitH);
      cursor += 0.11f;
      addNumber(static_cast<int>(kFuelCarryMax), cursor, textY, digitW, digitH, spacing);
      const std::string plutoniumText = std::to_string(playerState.carriedPlutonium);
      const float plutoniumDigitW = digitW * 0.9f;
      const float plutoniumDigitH = digitH * 0.9f;
      const float plutoniumSpacing = spacing * 0.9f;
      const float plutoniumWidth =
          static_cast<float>(plutoniumText.size()) * plutoniumDigitW +
          static_cast<float>(std::max(0, static_cast<int>(plutoniumText.size()) - 1)) * plutoniumSpacing;
      float plutoniumCursor = 0.80f - plutoniumWidth;
      addNumber(playerState.carriedPlutonium, plutoniumCursor, textY, plutoniumDigitW, plutoniumDigitH, plutoniumSpacing);
      if (!fuelTextLines.empty()) {
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        static_cast<GLsizeiptr>(fuelTextLines.size() * sizeof(glm::vec3)),
                        fuelTextLines.data());
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.90f, 0.72f, 0.22f);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(fuelTextLines.size()));
      }

      if (satelliteFueled) {
        if (const auto bombsite = predictBombsiteImpact(state.world, playerState.satellite)) {
          const glm::vec4 clip = miniProjection * miniView * glm::vec4(bombsite->impactPos, 1.0f);
          if (std::abs(clip.w) > 0.0001f) {
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            if (ndc.z >= -1.0f && ndc.z <= 1.0f && ndc.x >= -1.15f && ndc.x <= 1.15f && ndc.y >= -1.15f && ndc.y <= 1.15f) {
              const glm::vec3 baseReticleColor = bombsite->hitForcefield
                                                     ? glm::vec3(0.36f, 1.0f, 0.42f)
                                                     : (playerIndex == 0 ? glm::vec3(1.0f, 0.78f, 0.20f)
                                                                         : glm::vec3(0.62f, 0.92f, 1.0f));

              std::vector<glm::vec3> reticleLines;
              reticleLines.reserve(96);
              const auto addReticleSegment = [&](glm::vec2 a, glm::vec2 b) {
                reticleLines.push_back(glm::vec3(a, 0.0f));
                reticleLines.push_back(glm::vec3(b, 0.0f));
              };

              const glm::vec2 center2D(ndc.x, ndc.y);
              const float ringRadius = 0.072f;
              const float tickOuter = 0.122f;
              const float tickInner = 0.090f;
              const float crossOuter = 0.050f;
              const float crossGap = 0.015f;
              const int ringSegments = 28;

              for (int i = 0; i < ringSegments; ++i) {
                const float a0 = (static_cast<float>(i) / static_cast<float>(ringSegments)) * glm::two_pi<float>();
                const float a1 = (static_cast<float>(i + 1) / static_cast<float>(ringSegments)) * glm::two_pi<float>();
                addReticleSegment(center2D + glm::vec2(std::cos(a0), std::sin(a0)) * ringRadius,
                                  center2D + glm::vec2(std::cos(a1), std::sin(a1)) * ringRadius);
              }

              addReticleSegment(center2D + glm::vec2(-tickOuter, 0.0f), center2D + glm::vec2(-tickInner, 0.0f));
              addReticleSegment(center2D + glm::vec2(tickInner, 0.0f), center2D + glm::vec2(tickOuter, 0.0f));
              addReticleSegment(center2D + glm::vec2(0.0f, -tickOuter), center2D + glm::vec2(0.0f, -tickInner));
              addReticleSegment(center2D + glm::vec2(0.0f, tickInner), center2D + glm::vec2(0.0f, tickOuter));
              addReticleSegment(center2D + glm::vec2(-crossOuter, 0.0f), center2D + glm::vec2(-crossGap, 0.0f));
              addReticleSegment(center2D + glm::vec2(crossGap, 0.0f), center2D + glm::vec2(crossOuter, 0.0f));
              addReticleSegment(center2D + glm::vec2(0.0f, -crossOuter), center2D + glm::vec2(0.0f, -crossGap));
              addReticleSegment(center2D + glm::vec2(0.0f, crossGap), center2D + glm::vec2(0.0f, crossOuter));

              constexpr int kDriftSamples = 7;
              glm::vec2 lastDriftPoint = center2D;
              bool hasLastDriftPoint = false;
              for (int sample = kDriftSamples; sample >= 1; --sample) {
                SatelliteState priorSatellite = playerState.satellite;
                priorSatellite.orbitPhase -= playerState.satellite.orbitSpeed * 0.14f * static_cast<float>(sample);
                const auto priorPrediction = predictBombsiteImpact(state.world, priorSatellite);
                if (!priorPrediction.has_value()) {
                  continue;
                }

                const glm::vec4 priorClip = miniProjection * miniView * glm::vec4(priorPrediction->impactPos, 1.0f);
                if (std::abs(priorClip.w) <= 0.0001f) {
                  continue;
                }

                const glm::vec3 priorNdc = glm::vec3(priorClip) / priorClip.w;
                if (priorNdc.z < -1.0f || priorNdc.z > 1.0f || priorNdc.x < -1.2f || priorNdc.x > 1.2f ||
                    priorNdc.y < -1.2f || priorNdc.y > 1.2f) {
                  continue;
                }

                const glm::vec2 driftPoint(priorNdc.x, priorNdc.y);
                if (hasLastDriftPoint) {
                  addReticleSegment(lastDriftPoint, driftPoint);
                }
                lastDriftPoint = driftPoint;
                hasLastDriftPoint = true;
              }

              if (!reticleLines.empty()) {
                glBufferSubData(GL_ARRAY_BUFFER, 0,
                                static_cast<GLsizeiptr>(reticleLines.size() * sizeof(glm::vec3)),
                                reticleLines.data());
                glUniform3f(glGetUniformLocation(colorProgram, "uColor"),
                            baseReticleColor.x, baseReticleColor.y, baseReticleColor.z);
                glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(reticleLines.size()));
              }

              const std::array<glm::vec3, 4> centerDot = {
                  glm::vec3(center2D.x - 0.010f, center2D.y, 0.0f),
                  glm::vec3(center2D.x + 0.010f, center2D.y, 0.0f),
                  glm::vec3(center2D.x, center2D.y - 0.010f, 0.0f),
                  glm::vec3(center2D.x, center2D.y + 0.010f, 0.0f),
              };
              glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(centerDot), centerDot.data());
              glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 1.0f, 0.96f, 0.90f);
              glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(centerDot.size()));
            }
          }
        }
      }

      if (!satelliteFueled) {
        std::array<glm::vec3, kSatelliteNoiseSegments * 2> noiseLines{};
        const int noiseTime = static_cast<int>(currentFrame * 28.0f) + playerIndex * 97;
        for (int i = 0; i < kSatelliteNoiseSegments; ++i) {
          const float x0 = hashNoise(noiseTime * 19 + i * 37, noiseTime * 7 + i * 13) * 1.92f - 0.96f;
          const float y0 = hashNoise(noiseTime * 11 + i * 17, noiseTime * 23 + i * 29) * 1.92f - 0.96f;
          const float dx = (hashNoise(noiseTime * 31 + i * 41, noiseTime * 5 + i * 47) - 0.5f) * 0.16f;
          const float dy = (hashNoise(noiseTime * 43 + i * 53, noiseTime * 3 + i * 59) - 0.5f) * 0.10f;
          noiseLines[static_cast<std::size_t>(i * 2)] = glm::vec3(x0, y0, 0.0f);
          noiseLines[static_cast<std::size_t>(i * 2 + 1)] = glm::vec3(x0 + dx, y0 + dy, 0.0f);
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(noiseLines), noiseLines.data());
        const glm::vec3 staticColor =
            playerIndex == 0 ? glm::vec3(0.96f, 0.56f, 0.24f) : glm::vec3(0.58f, 0.88f, 1.0f);
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), staticColor.x, staticColor.y, staticColor.z);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(noiseLines.size()));

        std::array<glm::vec3, 16> tearLines{};
        for (int i = 0; i < 8; ++i) {
          const float y = hashNoise(noiseTime * 61 + i * 17, noiseTime * 67 + i * 13) * 1.8f - 0.9f;
          tearLines[static_cast<std::size_t>(i * 2)] = glm::vec3(-0.96f, y, 0.0f);
          tearLines[static_cast<std::size_t>(i * 2 + 1)] = glm::vec3(0.96f, y, 0.0f);
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(tearLines), tearLines.data());
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.92f, 0.96f, 0.96f);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(tearLines.size()));
      }

      glViewport(viewportX, viewportY, viewportWidth, viewportHeight);
      glScissor(viewportX, viewportY, viewportWidth, viewportHeight);

      if (playerState.hoveredBlock.has_value()) {
        const glm::vec3 blockPos = blockToWorld(playerState.hoveredBlock->block) - glm::vec3(0.001f);
        const glm::mat4 outlineModel = glm::translate(glm::mat4(1.0f), blockPos);
        glUseProgram(outlineProgram);
        glUniformMatrix4fv(glGetUniformLocation(outlineProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(outlineModel));
        glUniformMatrix4fv(glGetUniformLocation(outlineProgram, "uView"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(outlineProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(projection));
        glm::vec3 outlineColor(1.0f, 0.92f, 0.72f);
        if (playerState.mining.active && sameBlock(playerState.mining.block, playerState.hoveredBlock->block)) {
          const float progress =
              std::clamp(playerState.mining.progress / miningDurationFor(playerState.mining.type), 0.0f, 1.0f);
          outlineColor = glm::mix(glm::vec3(0.96f, 0.72f, 0.24f), glm::vec3(1.0f, 0.25f, 0.10f), progress);
        }
        glUniform3f(glGetUniformLocation(outlineProgram, "uColor"), outlineColor.x, outlineColor.y, outlineColor.z);
        glDisable(GL_CULL_FACE);
        glDepthMask(GL_FALSE);
        glBindVertexArray(outlineVao);
        glDrawArrays(GL_LINES, 0, 24);
        glDepthMask(GL_TRUE);
        glEnable(GL_CULL_FACE);

        const glm::ivec3 place = playerState.hoveredBlock->previous;
        bool canPlace = state.world.inBounds(place.x, place.y, place.z) && state.world.get(place.x, place.y, place.z) == Air;
        if (canPlace) {
          for (const PlayerState& otherPlayer : state.players) {
            if (playerIntersectsBlock(otherPlayer, otherPlayer.avatar.position, place)) {
              canPlace = false;
              break;
            }
          }
        }
        if (canPlace) {
          const glm::vec3 placePos = blockToWorld(place) - glm::vec3(0.001f);
          const glm::mat4 placeModel = glm::translate(glm::mat4(1.0f), placePos);
          glUniformMatrix4fv(glGetUniformLocation(outlineProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(placeModel));
          glm::vec3 placeColor(0.58f, 0.90f, 1.00f);
          if (playerState.selectedBlock == Ember) placeColor = glm::vec3(1.00f, 0.48f, 0.20f);
          else if (playerState.selectedBlock == DarkRock) placeColor = glm::vec3(0.74f, 0.72f, 0.88f);
          else if (playerState.selectedBlock == Crust) placeColor = glm::vec3(0.96f, 0.78f, 0.44f);
          if (playerState.placing.active && sameBlock(playerState.placing.block, place) &&
              playerState.placing.type == playerState.selectedBlock) {
            const float progress =
                std::clamp(playerState.placing.progress / placementDurationFor(playerState.placing.type), 0.0f, 1.0f);
            placeColor = glm::mix(placeColor * 0.72f, glm::vec3(1.0f, 0.96f, 0.82f), progress);
          }
          glUniform3f(glGetUniformLocation(outlineProgram, "uColor"), placeColor.x, placeColor.y, placeColor.z);
          glDepthMask(GL_FALSE);
          glBindVertexArray(outlineVao);
          glDrawArrays(GL_LINES, 0, 24);
          glDepthMask(GL_TRUE);
        }
      }

      glUseProgram(colorProgram);
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uView"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
      glUniform1f(glGetUniformLocation(colorProgram, "uAlpha"), 1.0f);
      glBindVertexArray(orbitLineVao);
      glBindBuffer(GL_ARRAY_BUFFER, orbitLineVbo);
      glDisable(GL_DEPTH_TEST);
      glDepthMask(GL_FALSE);
      glDisable(GL_CULL_FACE);

      const bool targetLocked = playerState.hoveredBlock.has_value();
      const glm::vec3 crosshairColor =
          targetLocked ? glm::vec3(1.0f, 0.92f, 0.74f) : glm::vec3(0.78f, 0.62f, 0.52f);
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

      std::vector<glm::vec3> healthTextLines;
      const auto addHudLine = [&](glm::vec2 a, glm::vec2 b) {
        healthTextLines.push_back(glm::vec3(a, 0.0f));
        healthTextLines.push_back(glm::vec3(b, 0.0f));
      };
      const auto addHudDigit = [&](int digit, float x, float y, float w, float h) {
        const glm::vec2 topL(x, y), topR(x + w, y), midL(x, y - h * 0.5f), midR(x + w, y - h * 0.5f),
            botL(x, y - h), botR(x + w, y - h);
        const bool seg[10][7] = {
            {true, true, true, false, true, true, true}, {false, false, true, false, false, true, false},
            {true, false, true, true, true, false, true}, {true, false, true, true, false, true, true},
            {false, true, true, true, false, true, false}, {true, true, false, true, false, true, true},
            {true, true, false, true, true, true, true}, {true, false, true, false, false, true, false},
            {true, true, true, true, true, true, true}, {true, true, true, true, false, true, true},
        };
        if (seg[digit][0]) addHudLine(topL, topR);
        if (seg[digit][1]) addHudLine(topL, midL);
        if (seg[digit][2]) addHudLine(topR, midR);
        if (seg[digit][3]) addHudLine(midL, midR);
        if (seg[digit][4]) addHudLine(midL, botL);
        if (seg[digit][5]) addHudLine(midR, botR);
        if (seg[digit][6]) addHudLine(botL, botR);
      };
      const auto addHudSlash = [&](float x, float y, float w, float h) {
        addHudLine(glm::vec2(x, y - h), glm::vec2(x + w, y));
      };
      const auto addHudNumber = [&](int value, float& cursor, float y, float w, float h, float spacing) {
        const std::string text = std::to_string(value);
        for (char ch : text) {
          addHudDigit(ch - '0', cursor, y, w, h);
          cursor += w + spacing;
        }
      };
      float healthCursor = -0.92f;
      const float healthY = 0.90f;
      const float healthDigitW = 0.080f;
      const float healthDigitH = 0.120f;
      const float healthSpacing = 0.026f;
      addHudNumber(static_cast<int>(std::ceil(playerState.health)), healthCursor, healthY, healthDigitW, healthDigitH,
                   healthSpacing);
      addHudSlash(healthCursor + 0.008f, healthY, 0.055f, healthDigitH);
      healthCursor += 0.09f;
      addHudNumber(static_cast<int>(kPlayerMaxHealth), healthCursor, healthY, healthDigitW, healthDigitH,
                   healthSpacing);
      if (!healthTextLines.empty()) {
        const float healthRatio = std::clamp(playerState.health / kPlayerMaxHealth, 0.0f, 1.0f);
        glm::vec3 healthColor =
            glm::mix(glm::vec3(1.0f, 0.28f, 0.18f), glm::vec3(1.0f, 0.88f, 0.72f), healthRatio);
        healthColor = glm::mix(healthColor, glm::vec3(1.0f, 1.0f, 1.0f), playerState.damageFlash * 0.65f);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        static_cast<GLsizeiptr>(healthTextLines.size() * sizeof(glm::vec3)),
                        healthTextLines.data());
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), healthColor.x, healthColor.y, healthColor.z);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(healthTextLines.size()));
      }

      std::vector<glm::vec3> scoreLines;
      scoreLines.reserve(48);
      const auto addScoreLine = [&](glm::vec2 a, glm::vec2 b) {
        scoreLines.push_back(glm::vec3(a, 0.0f));
        scoreLines.push_back(glm::vec3(b, 0.0f));
      };
      const auto addScoreDigit = [&](int digit, float x, float y, float w, float h) {
        const glm::vec2 topL(x, y), topR(x + w, y), midL(x, y - h * 0.5f), midR(x + w, y - h * 0.5f),
            botL(x, y - h), botR(x + w, y - h);
        const bool seg[10][7] = {
            {true, true, true, false, true, true, true}, {false, false, true, false, false, true, false},
            {true, false, true, true, true, false, true}, {true, false, true, true, false, true, true},
            {false, true, true, true, false, true, false}, {true, true, false, true, false, true, true},
            {true, true, false, true, true, true, true}, {true, false, true, false, false, true, false},
            {true, true, true, true, true, true, true}, {true, true, true, true, false, true, true},
        };
        if (seg[digit][0]) addScoreLine(topL, topR);
        if (seg[digit][1]) addScoreLine(topL, midL);
        if (seg[digit][2]) addScoreLine(topR, midR);
        if (seg[digit][3]) addScoreLine(midL, midR);
        if (seg[digit][4]) addScoreLine(midL, botL);
        if (seg[digit][5]) addScoreLine(midR, botR);
        if (seg[digit][6]) addScoreLine(botL, botR);
      };
      const auto addScoreNumber = [&](int value, float& cursor, float y, float w, float h, float spacing) {
        const std::string text = std::to_string(value);
        for (char ch : text) {
          addScoreDigit(ch - '0', cursor, y, w, h);
          cursor += w + spacing;
        }
      };

      float scoreCursor = -0.92f;
      const float scoreY = -0.74f;
      const float scoreDigitW = 0.065f;
      const float scoreDigitH = 0.10f;
      const float scoreSpacing = 0.022f;
      addScoreNumber(state.scores[0], scoreCursor, scoreY, scoreDigitW, scoreDigitH, scoreSpacing);
      scoreCursor += 0.05f;
      addScoreNumber(state.scores[1], scoreCursor, scoreY, scoreDigitW, scoreDigitH, scoreSpacing);
      if (state.match.mode == GameMode::TurnBased) {
        scoreCursor += 0.09f;
        addScoreNumber(state.match.roundNumber, scoreCursor, scoreY, scoreDigitW * 0.85f, scoreDigitH * 0.85f,
                       scoreSpacing * 0.85f);
        addScoreLine(glm::vec2(scoreCursor + 0.01f, scoreY - scoreDigitH * 0.85f),
                     glm::vec2(scoreCursor + 0.06f, scoreY));
        scoreCursor += 0.085f;
        addScoreNumber(kTurnBasedRounds, scoreCursor, scoreY, scoreDigitW * 0.85f, scoreDigitH * 0.85f,
                       scoreSpacing * 0.85f);
        scoreCursor += 0.08f;
        addScoreNumber(static_cast<int>(std::ceil(state.match.roundTimeRemaining)), scoreCursor, scoreY,
                       scoreDigitW * 0.85f, scoreDigitH * 0.85f, scoreSpacing * 0.85f);
      }
      if (!scoreLines.empty()) {
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        static_cast<GLsizeiptr>(scoreLines.size() * sizeof(glm::vec3)),
                        scoreLines.data());
        glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 1.0f, 0.92f, 0.64f);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(scoreLines.size()));
      }

      glEnable(GL_CULL_FACE);
      glDepthMask(GL_TRUE);
      glEnable(GL_DEPTH_TEST);

      const float swingT = playerState.hand.swinging ? (playerState.hand.swingTime / kHandSwingDuration) : 0.0f;
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
      glUniform1f(glGetUniformLocation(colorProgram, "uAlpha"), 1.0f);
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
      if (playerState.selectedBlock == Ember) glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.82f, 0.28f, 0.14f);
      else if (playerState.selectedBlock == DarkRock) glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.28f, 0.16f, 0.14f);
      else glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.56f, 0.22f, 0.12f);
      glDrawArrays(GL_TRIANGLES, 0, 36);

      glUniformMatrix4fv(glGetUniformLocation(colorProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(handModel));
      glUniform3f(glGetUniformLocation(colorProgram, "uColor"), 0.86f, 0.68f, 0.56f);
      glDrawArrays(GL_TRIANGLES, 0, 36);

      glEnable(GL_CULL_FACE);
      glDepthMask(GL_TRUE);
      glEnable(GL_DEPTH_TEST);
      glDisable(GL_SCISSOR_TEST);
    };

    const int halfWidth = width / 2;
    renderPlayerViewport(0, 0, 0, halfWidth, height);
    renderPlayerViewport(1, halfWidth, 0, width - halfWidth, height);

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
  if (gHiddenCursor) {
    glfwDestroyCursor(gHiddenCursor);
    gHiddenCursor = nullptr;
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
