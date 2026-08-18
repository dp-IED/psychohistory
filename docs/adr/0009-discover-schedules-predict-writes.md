# Discover schedules; predict writes the forecast

Discover opens problems and must set resolution day. It does not write Claim or Justification. The predict tick (formerly due-today) writes those fields on live problems, including revisions.

**Considered options**: (1) mint a first wakeup Due on predict; (2) discover writes the full first forecast; (3) discover sets resolution day, predict writes forecasts while live. (3) is the rule. Split Due vs Y is superseded by ADR 0010.
