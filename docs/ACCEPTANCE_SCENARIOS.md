# Acceptance Scenarios

## Scenario 1: Chat message to created trade
Input:
- "Buy 100 MW tomorrow at $47 from Shell"

Expected:
- Candidate component extracts required fields
- Creates trade via `POST /api/trades`
- Retrieves trade via `GET /api/trades/{trade_id}`
- Trade is persisted and readable

## Scenario 2: Email with missing hub
Input:
- "Please sell 80 MW tomorrow at $51 to BP"

Expected:
- Candidate component applies configured default hub or asks clarification
- Does not crash on missing optional context
- Creates trade successfully when required fields are complete

## Scenario 3: Ambiguous free-form request
Input:
- "Maybe buy some power next week from Conoco"

Expected:
- Low confidence result
- Candidate component avoids blind booking
- Candidate component asks for missing quantity and price

## Scenario 4: API conflict handling
Input:
- Attempt to update a trade with invalid payload (for example `price_per_mwh = -1`)

Expected:
- Candidate component handles validation error (`422`) gracefully
- Presents clear corrective guidance to the user
