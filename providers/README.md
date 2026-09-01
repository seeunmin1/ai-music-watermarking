# Provider Catalog

Each JSON file describes one generative audio provider. The detection backend
loads these profiles dynamically, so adding a new provider should not require
changes to the core provenance decision logic.

Required fields:

- `provider_id`
- `display_name`
- `aliases`
- `products`
- `supported_media`
- `signals`
- `c2pa_issuers`
- optional `c2pa_identity` patterns for certificate subjects, issuers, and
  claim-generator strings
- `official_verification`
- `confidence_policy`
- `parser_adapter`

Claim policy:

- Official API, trusted C2PA signature validation, or registered Audiomark
  watermark matches may produce `verified`.
- C2PA manifests that match a provider profile but do not validate against a
  trusted signature are reported as `detected`, not `verified`.
- Local attribution models may produce only `probable`.
- Metadata aliases and product names produce only `unknown` hints.
