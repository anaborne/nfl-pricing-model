This project is a computer model that estimates which NFL team will win a game, and a test
of those estimates against the odds the betting market offered. The model rates all 32
teams from past results, then prices each of the 272 games of the 2025 regular season using
only games played before it. Every prediction was scored against what happened and compared
with the market's price for the same game. The market was more accurate. On the standard
accuracy score for probability forecasts, where lower is better, the model scored 0.2239
and the market scored 0.2116, and a statistical test says that difference is unlikely to be
chance. The model was furthest off in the games where it disagreed with the market most,
which is the opposite of a useful betting signal. The repository reports this and gives the
reason. The model reads final scores and nothing else, while the market prices injuries,
weather and everything else the public knows. Anyone can download the code and run three
commands to get exactly these numbers back.
