% validity_check.pl
valid_sentence([please, help, me]).
valid_sentence([i, love, you]).
valid_sentence([i, need, you]).
valid_sentence([i, want, food]).
valid_sentence([go, home]).
valid_sentence([i, need, help]).
valid_sentence([please, stop]).
valid_sentence([yes, please]).
valid_sentence([no, please]).
valid_sentence([i, drink, water]).

is_valid_sentence(Sentence) :-
    valid_sentence(Sentence).

