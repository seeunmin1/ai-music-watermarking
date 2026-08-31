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
- `official_verification`
- `confidence_policy`
- `parser_adapter`

Claim policy:

- Official API or trusted C2PA/watermark matches may produce `verified`.
- Local attribution models may produce only `probable`.
- Metadata aliases and product names produce only `unknown` hints.
