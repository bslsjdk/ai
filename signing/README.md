# AIMENG debug signing key

This directory contains a **public test-only signing key** for the isolated
`bslsjdk.ornithnpu.aimengdebug` Android package.

- It is intentionally public so CI can produce repeatable, update-compatible test APKs without requiring users to configure GitHub Actions secrets.
- The private key and passwords are public. Anyone can sign a replacement APK for this test package. Do not trust APKs from unverified sources.
- Never use this key for production, Google Play, the main Ornith app, or MCNPU.
- The package suffix `.aimengdebug` keeps this test app separate from the regular `bslsjdk.ornithnpu` app.
- This key establishes stable signatures for future builds. APKs previously built with ephemeral runner-generated debug keys may require a one-time uninstall before installing the first APK signed with this key.

The workflow uses the alias and passwords directly; no GitHub Actions secrets are required.
