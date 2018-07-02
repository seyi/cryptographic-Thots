%% @author Seyi Akadri
%%  Provides API for the creation,query of blum integers.


-module(blum_integers).
-import(crypto_thots,[jacobi/2,is_composite/1,is_odd/1,legendre/2,split/1,debug/2]).

-define(GEN_SEED,1000000).
%% ====================================================================
%% API functions
%% ====================================================================
-export([sq_tab/3,sq_mod_query/3,sq_mod_query/6,is_qr/2,is_blum/1,bl_gen/2,qr/1]).




%% ====================================================================
%% Internal functions
%% ====================================================================

%% @doc generates a table of blum integers
-spec sq_tab(P,From,To) -> Integerlist when
		  P				:: integer(),
		  From 			:: integer(),
		  To				:: integer(),
		  Integerlist	:: [integer()].
sq_tab(P,From,To) ->
	[ {X, round(math:pow(X,2)) rem P} || X <- lists:seq(From,To)].

%% doc A naive implementation of creating a table of blum integers.

-spec sq_mod_query(Naive,Value,Prime,From,To,Limit) -> {N,SRes} when
		  Naive		:: atom(),
		  Value		:: integer(),
		  Prime		:: integer(),
		  From		:: integer(),
		  To			:: integer(),
		  Limit		:: integer(),
		  N 			:: integer(),
		  SRes		:: integer().
sq_mod_query(naive,Value,P, From,To,Limit) ->
	sq_q_helper(Value,P,From,To+?GEN_SEED,Limit,false).
sq_mod_query(naive,Value, P) ->
   Table  = sq_tab(P,1,100000000),
   lists:keysearch(Value,2,Table).
	%sq_q_helper(Value,P,90000000,90000000+25000,100000000,false).



sq_q_helper(Value,P,InitialFrom,InitialTo,Limit,Res) when Res == false , InitialFrom < Limit->
	Table = sq_tab(P,InitialFrom,InitialTo),
	case lists:keysearch(Value,2,Table) of
		false ->  sq_q_helper(Value,P,InitialFrom+1,InitialTo+?GEN_SEED+1,Limit,false);
		{value,{N,SRes}} -> 
			sq_q_helper(Value,P,InitialFrom,InitialTo,Limit,{N,SRes})
    end	
	;


sq_q_helper(_,_,_,_,_,Res)  ->
	Res.

%% @doc Queries  number whether it is square modulo a prime
-spec is_sq_mod(A,N) -> {Ok,True} when
		  A		:: integer(),
		  N		:: integer(),
		  Ok		:: atom(),
		  True	:: boolean().
is_sq_mod(A,P) ->
	Table = sq_mod_query(naive,A, P),
	case lists:keysearch(A,2,Table) of
		false ->  not_found;
		{value,{N,SRes}} -> 
			{ok,true}
    end	
	.

%% @doc Checks whether the Number is a quadratic residuo of N

-spec is_qr(A,N) -> {Ok,Bool} when
	A		:: integer(),
	N		:: integer(),
	Ok		:: atom(),
	Bool		:: boolean().

is_qr(A,N) ->
 case is_composite(N)  of
	true -> 
		case is_odd(N) of 
			{ok,true} ->
				case split(N) of 
					[{F1,F2} | Rest] ->
					  L1 = jacobi(A,F1), L2 =  jacobi(A,F2),
					  debug("factorization of N : l(~p,~p),l(~p,~p)~n",[F1,N,F2,N]),
					  %{atom_to_list(legendre)++" "++(integer_to_list(F1)++"/"++integer_to_list(N)),
                       % atom_to_list(legendre)++" "++(integer_to_list(F2)++"/"++integer_to_list(N))
						% ,legendre(F1,N) , legendre(F2,N)}; 
						case L1 * L2 == 1 of true -> {ok,true};false -> {ok,false} end	;
					[] -> 
 						error(integer_to_list(N)++ " not yet supported reason : too large")
				end;
						
			{ok,false} ->
				error(bad_argument)
					
		end ;
	false -> 
		case jacobi(A,N) of
				1 -> {ok,true};
				-1 -> {ok,false};
				_ -> {ok,no_solution}
		end		
 end
	.

%% @doc Checks whether the argument is a blum integer.
%% @param N 
-spec is_blum(N) -> {Ok,Bool} when
	N 	:: integer(),
	Ok	:: atom(),
	Bool	:: boolean().
is_blum(N) ->
	case   split(N) of 
		[{F1,F2} | Rest] -> 
		  case not (is_composite(F1)) andalso not (is_composite(F2)) andalso F1 /= F2 of
		  	true ->
				case (F1 rem 4 == 3 ) andalso (F2 rem 4 == 3) of
					true -> {ok,true};
					false -> {ok,false}
			    end;
			false -> {ok,false}
		  end;
			
		[] -> {ok,false}
	end.
	

bl_gen(From,To) ->
	[ X || X <- lists:seq(From,To), case is_blum(X) of {ok,true} -> true; {ok,false} -> false end ].

qr(N) ->
	[ X || X <- lists:seq(1,N), case is_qr(X,N) of {ok,true} -> true; {ok, false} -> false; {ok,no_solution} -> false end].
	

