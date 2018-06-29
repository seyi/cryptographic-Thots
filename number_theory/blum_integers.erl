%% @author anointedone
%% @doc @todo Add description to blum_integers.


-module(blum_integers).

-define(GEN_SEED,25000).
%% ====================================================================
%% API functions
%% ====================================================================
-export([sq_tab/3,sq_mod_query/3,sq_mod_query/6]).




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

sq_mod_query(naive,Value,P, From,To,Limit) ->
	sq_q_helper(Value,P,From,To+?GEN_SEED,Limit,false).
sq_mod_query(naive,Value, P) ->
   Table  = sq_tab(P,1,100000000),
   lists:keysearch(Value,2,Table).
	%sq_q_helper(Value,P,90000000,90000000+25000,100000000,false).



sq_q_helper(Value,P,InitialFrom,InitialTo,Limit,Res) when Res == false ->
	Table = sq_tab(P,InitialFrom,InitialTo),
	case lists:keysearch(Value,2,Table) of
		false ->  sq_q_helper(Value,P,InitialFrom+1,InitialTo+?GEN_SEED+1,Limit,false);
		{value,{N,SRes}} -> 
			sq_q_helper(Value,P,InitialFrom,InitialTo,Limit,{N,SRes})
    end	
	;

sq_q_helper(_,_,_,_,_,Res)  ->
	Res.

