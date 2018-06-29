%% @author anointedone
%% @doc @todo Add description to blum_integers.


-module(blum_integers).

%% ====================================================================
%% API functions
%% ====================================================================
-export([sq_tab/3]).



%% ====================================================================
%% Internal functions
%% ====================================================================

-spec sq_tab(P,From,To) -> Integerlist when
		  P				:: integer(),
		  From 			:: integer(),
		  To				:: integer(),
		  Integerlist	:: [integer()].
sq_tab(P,From,To) ->
	[ {X, round(math:pow(X,2)) rem P} || X <- lists:seq(From,To)].

