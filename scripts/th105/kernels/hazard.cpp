#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#ifdef _WIN32
#define TH105_EXPORT extern "C" __declspec(dllexport)
#else
#define TH105_EXPORT extern "C"
#endif

namespace {

struct Projectile {
    float x;
    float y;
    float velocityX;
    float velocityY;
    float halfWidth;
    float halfHeight;
    float accelerationX;
    float accelerationY;
};

struct Candidate {
    float velocityX;
    float velocityY;
    float halfWidth;
    float halfHeight;
    std::int32_t grazeFrames;
    std::int32_t startupFrames;
};

struct RiskResult {
    std::int32_t safe;
    std::int32_t firstCollisionFrame;
    float minimumClearance;
    float finalX;
    float finalY;
};

float signedAabbClearance(
    float playerX,
    float playerY,
    float playerHalfWidth,
    float playerHalfHeight,
    float hazardX,
    float hazardY,
    float hazardHalfWidth,
    float hazardHalfHeight
) {
    const float gapX = std::fabs(playerX - hazardX)
        - playerHalfWidth - hazardHalfWidth;
    const float gapY = std::fabs(playerY - hazardY)
        - playerHalfHeight - hazardHalfHeight;
    if (gapX <= 0.0F && gapY <= 0.0F) return std::max(gapX, gapY);
    return std::hypot(std::max(0.0F, gapX), std::max(0.0F, gapY));
}

bool finiteProjectile(const Projectile& projectile) {
    return std::isfinite(projectile.x)
        && std::isfinite(projectile.y)
        && std::isfinite(projectile.velocityX)
        && std::isfinite(projectile.velocityY)
        && std::isfinite(projectile.halfWidth)
        && std::isfinite(projectile.halfHeight)
        && std::isfinite(projectile.accelerationX)
        && std::isfinite(projectile.accelerationY)
        && projectile.halfWidth >= 0.0F
        && projectile.halfHeight >= 0.0F;
}

}  // namespace

TH105_EXPORT std::int32_t th105_hazard_abi_version() {
    return 4;
}

TH105_EXPORT std::int32_t th105_evaluate_linear_paths(
    float playerX,
    float playerY,
    float playerHalfWidth,
    float playerHalfHeight,
    float collisionMargin,
    std::int32_t horizon,
    const Projectile* projectiles,
    std::uint32_t projectileCount,
    const Candidate* candidates,
    std::uint32_t candidateCount,
    RiskResult* results
) {
    if (!std::isfinite(playerX) || !std::isfinite(playerY)
        || !std::isfinite(playerHalfWidth) || !std::isfinite(playerHalfHeight)
        || !std::isfinite(collisionMargin)
        || playerHalfWidth < 0.0F || playerHalfHeight < 0.0F
        || horizon <= 0 || horizon > 600
        || projectileCount > 1024 || candidateCount == 0 || candidateCount > 64
        || (projectileCount && projectiles == nullptr)
        || candidates == nullptr || results == nullptr) {
        return -1;
    }
    for (std::uint32_t index = 0; index < projectileCount; ++index) {
        if (!finiteProjectile(projectiles[index])) return -2;
    }
    for (std::uint32_t candidateIndex = 0; candidateIndex < candidateCount;
         ++candidateIndex) {
        const Candidate candidate = candidates[candidateIndex];
        if (!std::isfinite(candidate.velocityX)
            || !std::isfinite(candidate.velocityY)
            || !std::isfinite(candidate.halfWidth)
            || !std::isfinite(candidate.halfHeight)
            || candidate.halfWidth < 0.0F || candidate.halfHeight < 0.0F
            || candidate.grazeFrames < 0 || candidate.grazeFrames > horizon
            || candidate.startupFrames < 0 || candidate.startupFrames > horizon) return -3;
        const float candidateHalfWidth = candidate.halfWidth > 0.0F
            ? candidate.halfWidth : playerHalfWidth;
        const float candidateHalfHeight = candidate.halfHeight > 0.0F
            ? candidate.halfHeight : playerHalfHeight;
        RiskResult result{
            1,
            -1,
            std::numeric_limits<float>::infinity(),
            playerX + candidate.velocityX * static_cast<float>(horizon - candidate.startupFrames),
            playerY + candidate.velocityY * static_cast<float>(horizon - candidate.startupFrames),
        };
        for (std::int32_t frame = 1; frame <= horizon; ++frame) {
            const std::int32_t movementFrame = std::max(0, frame - candidate.startupFrames);
            const float x = playerX + candidate.velocityX * static_cast<float>(movementFrame);
            const float y = playerY + candidate.velocityY * static_cast<float>(movementFrame);
            if (frame > candidate.startupFrames
                && frame <= candidate.startupFrames + candidate.grazeFrames) continue;
            for (std::uint32_t projectileIndex = 0;
                 projectileIndex < projectileCount; ++projectileIndex) {
                const Projectile projectile = projectiles[projectileIndex];
                const float clearance = signedAabbClearance(
                    x,
                    y,
                    candidateHalfWidth,
                    candidateHalfHeight,
                    projectile.x + projectile.velocityX * static_cast<float>(frame)
                        + 0.5F * projectile.accelerationX
                            * static_cast<float>(frame * frame),
                    projectile.y + projectile.velocityY * static_cast<float>(frame)
                        + 0.5F * projectile.accelerationY
                            * static_cast<float>(frame * frame),
                    projectile.halfWidth,
                    projectile.halfHeight
                );
                result.minimumClearance = std::min(result.minimumClearance, clearance);
                if (clearance <= collisionMargin && result.firstCollisionFrame < 0) {
                    result.safe = 0;
                    result.firstCollisionFrame = frame;
                }
            }
        }
        results[candidateIndex] = result;
    }
    return static_cast<std::int32_t>(candidateCount);
}
