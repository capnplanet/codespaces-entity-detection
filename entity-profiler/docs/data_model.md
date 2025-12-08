# Data Model

- **Observation**
  - `entity_id`
  - `timestamp`
  - `camera_id`
  - `fused_features`

- **EntityProfile**
  - `entity_id`
  - `observations: List[Observation]`

Entities are pseudonymous and may later be mapped to real-world identities using external evidence.
