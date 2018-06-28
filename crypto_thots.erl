%% @author seyi akadri
%% @doc @todo Add description to crypto_thots.


-module(crypto_thots).

%% ====================================================================
%% API functions
%% ====================================================================
-export([split/1,test_split/1,is_composite/1,
		 prime_factorization/1,pf2/1,pf3/1,
		 hd_fac_is_prime/1,test_prime_facts/1,test_perfect_square/1,
		 generate/2,test_generate/2,
		 test_search/3,inner_search_helper/3,search/3,
		 test_perfect_powers/2,is_perfect_power/1,
		 test_gcd/3,test_is_relative_prime/2,
		 test_check_similarity/2,call_relative_prime/2,
		 test_gcd/2,test_least_common_multiple/2,
		 test_least_common_multiple/1,multiply/4,
		 modular_inverse/3,extended_euclid/2,verify/4,convert2/2,
		 modular_exponent/4,is_even/1,is_odd/1,jacobi/2,
		 verify/2,quick_test/4,numeric_rep/2,div_till_odd/1,jacobi_helper/3,pow_bin/2,
		 legendre/2]
		).
-define(NOT_YET_IMPLEMENTED,not_yet_implemented).
-define(V_FAIL,verify_failure).
-define(V_PASS,verify_passed).
-define(JACOBI_BAD_ARG,bad_argument).
-define(LEGENDRE_BAD_ARG,bad_argument).
-define(UTILITY_CODE,"%%Utility code
%% TODO: TO BE IN SEPERATE MODULE").
-define(UNDEFINED_PROPERTY_CONDITION,"Condition not defined for property ~p~n").
-record(crypto_system,{type}).
-record(intractable_cpu_problems,{factoring,rsap,qrp,sqroot,
								  dlp,gdlp,dhp,gdhp,subset_sum}).
-record(ext_euc_result,{d,x,y}).
-record(ext_euc_args,{x1,x2,y1,y2}).
-record(jacobi_prop, {m,n,a,b,results_spec=[0,1,-1],result::jacobi_result(),gcd_a_n,
					  numer_list,denom_list}).

-type odd_prime() :: odd_prime().
-type odd_number() :: odd_number().
-type jacobi_result() :: jacobi_result().



%% ====================================================================
%% Internal functions
%% ====================================================================
solve(Computational_problem) ->
	error("Not yet Unimplemented").
determine_intractibility(Computational_problem) ->
	case solve(Computational_problem) of
		{ok,polytime} -> easy;
		{ok,exponential_time} -> intractable
	end.


algorithm_solution(Computation_problem) ->
	error("Not yet implemented").
poly_time_reduce(A,B) ->
	case algorithm_solution(A) of
		{ok,uses_b_subroutine,runs_in_polytime} -> true;
		{_,_,_} -> false
	end.

is_intractable(B,A) ->
	case poly_time_reduce(A,B) of
		true -> true;
		false -> false
	end.
%%def 3.2.
computational_equivalent(A,B) ->
	case (poly_time_reduce(A,B)) and (poly_time_reduce(B,A))  of
		true -> {ok,true};
			
		false ->{ok,false}
    end.


find_prime_factorization(Number) ->
	error("Not yet implemented").
	
%%Def 3.3. given a positive integer, n, find it's prime factorization,
%% that is write n = psub1expe1 ... psubkexpk where the psubiare pairwise,
%%distinct primes and each expi >= 1
%% remark 3.4 (primality versus factoring) ...beforeattempting to factor
%% an integer N, the integer should be tested to make sure that it is
%% indeed composite
integer_factorization_problem(N) ->
	Primes = find_prime_factorization(N).
	
%%Remark 3.5..Non trivial factorization of N of the from
%% n = ab , where 1 < a < n and a < b < n;
%% a and b are said to be no-trivial factors of n.
%% Here a and b are not necessariliy prime

split(N) ->
	NRoot = round(math:sqrt(N)),
	split(N,NRoot,2,[]).
		

split(N, Nroot,Countup,Acc) when (Countup  =< Nroot) and (Countup > 1) ->
	case (N rem Countup =:= 0) of
		true ->
			%%{A,B} = Countup
			split(N,Nroot,Countup+1,[{Countup,round(N/Countup)} | Acc]);
		false -> 
			split(N,Nroot,Countup+1,Acc)
	end;
split(_,_,_,Acc) ->
	lists:reverse(Acc).

test_split(N) ->
  [ (io:format("~p~p ~n" ,[X,split(X)])) || X <- lists:seq(1,N) ].

-spec is_composite(N) -> Bool when
	N	:: integer() ,
	Bool :: boolean().
is_composite(N) ->
	case split(N) of 
		[] -> false;
	     _ -> true
   end.	

% [ {case crypto_thots:is_composite(X) of true -> crypto_thots:split(X); false -> X end ,case crypto_thots:is_composite(Y) of true -> crypto_thots:split(Y); false -> Y end}  ||  {X,Y}  <- crypto_thots:split(54) ]. 
prime_factorization(N) ->
	%Trivial_factors = split(N),
	prime_factorization(N,[]).
prime_factorization([{X,Y}] = TfacHead,Acc) when length(TfacHead) > 0 ->
  	%{X,Y} = hd(TfacHead),
	%% hd(split(Y))
	%io:format("args passed ~p~n",[TfacHead]),
	case is_composite(Y) of
		true ->
			%io:format("~p is composite ~n",[Y]),
			%io:format("~p is not composite ~n",[X]),
			%io:format("~p TfacHead ~n",[[{X,Y}]]),
             {X1,Y1} = hd(split(Y)),
			 %Acc1 = [X | Acc];
			%is_composite;
		    prime_factorization([hd(split(Y))],[X | Acc])  ;
			%io:format("~p current accumulator value ~n",[Acc1]);
			%[X | prime_factorization(hd(split(Y)),Acc) ] ;
		false ->
			%io:format("false occured at ~p ~p~n",[X,Y]),
			[X | prime_factorization([(split(Y))],[Y | Acc])];
			
		_ ->
			io:format("shouldnt get here?"),
			shouldnt_get_here
	end;


prime_factorization(T,Acc) -> 
	%io:format("end of loop when T is ~p~n",[T]),
							  lists:reverse(Acc).

pf2(N) ->
	[ {case crypto_thots:is_composite(X) of
		   true -> crypto_thots:split(X); 
		   false -> X end ,
	   case crypto_thots:is_composite(Y) of 
		   true -> crypto_thots:split(Y); 
		   false -> Y end}  ||  {X,Y}  <- crypto_thots:split(N) ].

%% call to local/imported function is_composite/1 is illegal in guard
%% pf3(N) when is_composite(N) ->
%% 	is_composite;
%% pf3(_) ->
%% 	not_composite.
-type prime_nos() :: prime_nos().
-spec prime_nos() -> Number when
		  Number ::prime_nos().
prime_nos() ->
	error("Undefined function").
-type trivial_factors() :: [number()].

	

-spec pf3(N) -> [prime_nos()] when
		  N::[trivial_factors()]. 
		  
pf3(N) when length(N) > 0 ->
	{X,_} = hd(N);
	
pf3(_) ->
	composite.

%%  lists:foreach(fun(N) -> 
%%      io:format("~p is composite? ~p~n",[N,crypto_thots:pf3([ (X5) || X5<- crypto_thots:split(N) ] )]) end,lists:seq(1,100)) .

%% prove_that_first factor will be a prime
hd_fac_is_prime(N) ->
	lists:foreach(fun(N) -> 
      io:format("~p is composite? ~p~n",
				[N,crypto_thots:pf3([ (X5) || X5<- crypto_thots:split(N) ] )]) end,lists:seq(1,100)) .

%% crypto_thots:test_prime_facts(10).
test_prime_facts(N) ->
   [ io:format("At ~p pfac are ~p~n",[X,crypto_thots:prime_factorization([hd(crypto_thots:split(X))])]) || X <- lists:seq(1,N),crypto_thots:is_composite(X)].

%% 3.6 note Testing for perfect powers
%% if N >= 2 , testing for perfect power ie
%% N = (x)exp(k) for some integers x >= 2, k >= 2
%% for each p =< ln N   prime in range 2 to math:log2(N).    

%%, an integer approximation
%% x of {n}exp(1/p) is computed...
%% done by computing a binary search for x satisfying
%% n = {x}exp(p) in the interval 
%% [2, {2}exp((log N/P) +1)] math:pow(2,(math:log(64/2) ) + 1).

%%  math:pow(2,(math:log(64/2) ) + 1).
%% 22.09634588208995
%% 33> math:pow(2,(math:log(36/2) ) + 1).
%%14.829229737434863

is_perfect_power(N) when N >= 2 ->
	search(x_exp_prime,N,generate(primes_less_number_log_n,N));
is_perfect_power(_) ->
	error("bad argument").

%simple search, binary search will be more efficient of course
%search(x_exp_prime,N,X_Interval,Primes) ->
%	search(x_exp_prime,N,X_Interval,Primes);


search(x_exp_prime,N,[P|PT]) ->
	case inner_search_helper(generate(x_interval_for_prime,{N,P}
								),N,P) of
		{true,X,Exp} ->
			{true,X,Exp};
		not_found ->
			search(x_exp_prime,N,PT)
	end;
			
		

search(x_exp_prime,N,[]) -> 
	false.
test_search(x_exp_prime,N,Primes) ->
	search(x_exp_prime,N,Primes).

inner_search_helper([X|XT],N,P) ->
	case (floor(math:pow(X,P))  ) =:= N  of
		
		true -> {true,X,P};
		false -> inner_search_helper(XT,N,P)
	end;

inner_search_helper([],N,P) -> not_found.

test_inner_search_helper(N,P) ->
	crypto_thots:inner_search_helper(crypto_thots:generate(x_interval_for_prime,{N,P}),N,P).

test_perfect_powers(From,To) ->
	[ lists:map(fun(X) -> 
						case ( crypto_thots:search(x_exp_prime,X,crypto_thots:generate(primes_less_number_log_n,X))) of 
							   {Res,Base,Exp} -> io:format("~p is perfect power (~p raise to power ~p = ~p~n)",[X,Base,Exp,floor(math:pow(Base,Exp))]);
							   false -> ok end  end,[X]) || X<- lists:seq(From,To)  ].


	

	
% lists:seq(2,math:floor(Logn))
generate(primes_less_number_log_n,N) ->
	[X || X<- lists:seq(2,round(math:floor(math:log2(N)))),(not is_composite(X))];

generate(x_interval_for_prime,{N,P}) ->
	lists:seq(2,round(math:floor(math:pow(2,(math:log(N/P) ) + 1)))).	


test_generate(From,To) ->
	[io:format("N is ~p and primes less than logN are ~p~n",
			   [X,generate(primes_less_number_log_n,X)]) || X <-
	   lists:seq(From,To)].
	
test_perfect_square(N) ->
  [io:format("~p is a perfect square with root ~p~n",[X,(floor(math:sqrt(X)))]) || X <- lists:seq(1,N),
													  (math:sqrt(X) - (floor(math:sqrt(X)) + float(0.0001))) < float(0.0001),X > 1].


call_relative_prime(A,B) ->
	AS = split(A),
	BS = split(B),
	case AS of
		[ ] -> 
			case BS of
				[] -> true;
                 _ -> is_relative_prime([{1,A}],BS,[{1,A}],BS)
			end;
		_ -> 
			case BS of
				[] -> is_relative_prime(AS,[{1,B}],AS,[{1,B}]);
                 _ -> is_relative_prime(AS,BS,AS,BS)
			end	
	end.
	
			  
		 
	

is_relative_prime([D1H|D1T],[D2H|D2T],D1Acc,D2Acc) ->
	{AF1,AF2} = D1H,
	{BF1,BF2} = D2H,
	AFacs = [AF1,AF2],
	BFacs = [BF1,BF2],
	io:format("D1H => ~p~n", [D1H]),
	case check_similarity(D1H,D2H) of 
		true -> false;
		false -> is_relative_prime(D1H,D2T,D1T,D2Acc)
	end;

is_relative_prime([],_,_,_) -> true;


is_relative_prime({X,Y} = DElem,[D2H|D2T],D1TAcc,D2Acc) ->
	io:format("checking similarity of ~p and ~p ~n", 
				  [DElem,D2H]),
	case check_similarity(DElem,D2H) of
		
	 	  true -> false;
		   false -> is_relative_prime(DElem,D2T,D1TAcc,D2Acc)
	end;

is_relative_prime({X,Y} = DElem,[],D1TAcc,D2Acc) ->
	io:format("~p => DElem  D1TailAcc =>~p D2Accc => ~p~n", [DElem,D1TAcc,D2Acc]),
	is_relative_prime(D1TAcc,D2Acc,D1TAcc,D2Acc).





check_similarity({A,B},{C,D}) when
   (A =:= C) or (A =:= D) or (B =:= C) or (B =:= D)-> true;
check_similarity(_,_) -> false.

test_check_similarity(A,B) ->
	 check_similarity(A,B).

cf(A,B) ->
	R = [ X || X<-B ,lists:member(X,A)],
	case R of
		[] -> false;
		_ -> {ok,{A,B}}
	end.

 test_is_relative_prime(A,B) ->
%% 	{J,K} = hd(split(A)),
%% 	{L,M} = hd(split(B)),
%% 	is_relative_prime([J,K],[L,M]).
    hello.
gcd(euclid, X, Y) when X > 0 ->
	gcd(euclid,Y rem X, X);
gcd(euclid,X,Y) when X =:= 0 ->
	Y.

test_gcd(euclid,X,Y) ->
	gcd(euclid,X,Y).

gcd(euclid,[H|T]) ->
	gcd(euclid,H,gcd(euclid,T));
  
gcd(euclid,[]) -> 0.

test_gcd(euclid,L) ->
	gcd(euclid,L).

least_common_multiple(X,Y) when X > 0 ->
	round((X*Y)/gcd(euclid,X,Y));

least_common_multiple(X,Y) when X =:= 0 ->
	error("bad argument").

test_least_common_multiple(X,Y) ->
	least_common_multiple(X,Y).


%% crypto_thots:test_least_common_multiple([2, 5, 6, 1, 9]).
%% 0
%% 103> Z1 = 2*5 /crypto_thots:test_gcd(euclid,2,5).             
%% 10.0
%% 104>  10*6 /crypto_thots:test_gcd(euclid,10,6).  
%% 30.0
%% 105>  30*9 /crypto_thots:test_gcd(euclid,30,9).
%% 90.0



 least_common_multiple(L) ->
	
	lists:foldl(fun(A,Acc) -> crypto_thots:test_least_common_multiple(A,Acc) end,hd(L),L).
	

 
 test_least_common_multiple(L) ->
 	least_common_multiple(L).


extended_euclid(A,B) ->
	extended_euclid(A,B,1,0,0,1).
extended_euclid(A,B,X2,X1,Y2,Y1) when B > 0 ->
	Q  = A div  B,
	R = A - (Q*B),
	X = X2 -(Q*X1),
	Y = Y2 -(Q*Y1),
	io:format("A:~p, B:~p, X2:~p,X1:~p,Y2:~p,Y1:~p~n", [A,B,X2,X1,Y2,Y1]),
	extended_euclid(B,R,X1,X,Y1,Y);

	  
extended_euclid(A,B,X2,X1,Y2,Y1) when B =:= 0 ->
		%d=A,x=1,y=0 return {d,x,a}
		
		#ext_euc_result{d=A,x=X2,y=Y2};



extended_euclid(A,B,_,_,_,_)  when B > A-> 
	{error,"Bad Argument"};

extended_euclid(A,_,X2,X1,Y2,Y1) ->
	#ext_euc_result{d=A,x=X2,y=Y2}.
 
verify(ext_euclid,A,B,Res) when is_record(Res,ext_euc_result) ->
	
	#ext_euc_result{ d = D,x = X,y = Y}   = Res,D,X,Y,
	case  ((gcd(euclid,A,B)) =:= D) andalso ((A*X) + (B*Y) =:= D) of
		true -> {ok,passed};
		false -> {assertion_failed}
	end;

verify(ext_euclid,_,_,_) ->
	error("bad argument").



multiply(mod,A,B,ModC) ->
	P1 = (A rem ModC) ,
	P2 = (B rem ModC),
	(P1 * P2) rem ModC.


modular_inverse(naive,A,ModC) ->
	modular_inverse(naive,A,0,ModC);
modular_inverse(ext_euclid,A,B) ->
	Result = extended_euclid(A,B,1,0,0,1),
    case Result#ext_euc_result.d /= 1 of
			true -> inverse_does_not_occur;
			false -> {ok,Result#ext_euc_result.x}
	end.

modular_inverse(naive,A,Counter,ModC) when Counter =< (ModC-1)->
	case multiply(mod,A,Counter,ModC) of
       1 -> Counter ;
	   _ -> modular_inverse(naive,A,Counter+1,ModC)
	end;

modular_inverse(naive,A,Counter,ModC) when Counter =:= (ModC) ->
	no_inverse.


%%Repeated square and multiply algorithm

-spec modular_exponent(RSMA,A,K,ModN) -> Integer when
		  RSMA :: atom(),
		  A :: integer(),
		  K :: integer(),
		  ModN :: integer(),
		  Integer ::integer().
modular_exponent(rsma,A,K,ModN) when K =:= 0 ->
	modular_exponent(rsma,A,[],1,1,ModN);
modular_exponent(rsma,A,K,ModN) ->
	modular_exponent(rsma,A,convert2(base2,K),1,1,ModN).


	
modular_exponent(rsma,A,KbinList,Counter,B, N) 
  						when Counter =< length(KbinList)  ->
	%io:format("A val = ~p B val ~p K = ~p~n", [A,B,lists:nth(Counter, KbinList)]),
	case (Counter == 1) andalso lists:nth(1, KbinList) =:= 1 of
		true -> modular_exponent(rsma,A,KbinList,Counter+1,A,N);
		false ->
			
			Acomp = (round(math:pow(A,2)) rem N),
			case (lists:nth(Counter, KbinList) =:= 1) of
				true -> 
					Bcomp = A * B rem N,
					
					modular_exponent(rsma,Acomp,KbinList,Counter+1,Bcomp,N);
				false -> 
					
					modular_exponent(rsma,Acomp,KbinList,Counter+1,B,N)
			end
	end;
			
	
modular_exponent(rsma,_,K,_,_,_) when (length(K) == 0) -> 
	1;

modular_exponent(rsma,_,_,_,B,_) ->
	B.

-spec jacobi_compute(A,P) -> Number when
		  A :: integer(),
		  P :: odd_prime(),
		  Number :: integer().
jacobi_compute(A,P) ->
	0.


%%Lengendre
-spec legendre(A,N) -> Number when
		  A 			:: integer(),
		  N			:: odd_prime(),
		  Number	   	:: integer().
legendre(A,N) ->
	legendre(A,N,1).
legendre(A,P,Acc) when A =:= 1 ->
	1;
legendre(A,P,Acc) when A =:= 0 ->
	0;
legendre(A,P,Acc) when A =:= -1 ->
	case  (is_composite(P)) of 
		true -> error(bad_legendre_arg);
		false -> 
			case P rem 4 of
				1 -> 1 * Acc;
				3 -> -1 * Acc;
				_ -> {bad_argument,P}
			end
	end;
	
legendre(A,P,Acc) when A  =:= P ->
	0;

legendre(A,P,Acc) when A =:= 2 ->
		debug("~pl(~p,~p)~n",[Acc,A,P]),
		case is_even(P) of 
		{ok,true} ->{?JACOBI_BAD_ARG,"N cannot be even"} ;
		{ok,false} ->
			case (P rem 8) of
				1 -> 1 * Acc;
				7 ->  1 * Acc;
				3 -> -1 * Acc;
				5 -> -1 * Acc;
			    _ -> {?JACOBI_BAD_ARG,"legendre property for A",[A,P]}
			end
	end;

legendre(A,N,Acc) when is_integer(A),is_integer(N) ->
	debug("~pl(~p,~p)~n",[Acc,A,N]),
	case is_even(A) of
		{ok,true} ->
			{OddNum,Exp} = div_till_odd(A),
			%debug("(legendre(~p/~p))^~p , legendre(~p,~p)~n",[2,N,Exp,OddNum,N]),
			Twores = legendre(2,N,Acc),
			legendre(OddNum,N,round(math:pow(Twores,Exp))*Acc);
		{ok,false} ->
 			case A > N  of
				true ->
					Decomp = A rem N,
					%io:format("Decomp = ~p~n",[Decomp]),
  					%debug("(legendre(~p/~p)~n",[Decomp,N]),
 					legendre(Decomp,N,Acc);					
				false ->
					case is_composite(A) of
						true -> case hd(split(A)) of {F1,F2} -> io:format("l(~p,~p) * l(~p,~p) ~n",[F1,N,F2,N]),
																legendre(F1,N,Acc *legendre(F2,N,Acc)); [] -> done end  ;
						false -> 
							Flip_res = jacobi_flip(A,N),
 							debug("(~p)legendre(~p/~p)~n",[Flip_res,N,A]),
  							legendre(N,A,Flip_res* Acc)
					end
					 
			end
	end;
 

legendre(_,_,Acc)  -> Acc.


				   

%%Jacobi Properties
-spec jacobi(Args,N) -> Number when
		Args ::[number()] | integer(),
		N    :: [odd_number()] | odd_number(),
		Number :: jacobi_result.

jacobi(A,N) when is_list(N) andalso is_integer(A) ->
		3;
jacobi(Args,N) when is_list(Args) andalso is_integer(N) -> 
		4;
jacobi(Args,N) when is_list(Args) andalso is_list(N) -> 
 5;

jacobi(A,N)  ->
	jacobi_helper(A,N,1).


jacobi_helper(A,_,Acc) when A == 0 ->
	0 ;
jacobi_helper(A,_,Acc) when A == 1 ->
	1 * Acc;
jacobi_helper(A,N,Acc) when A == N ->
	0;

jacobi_helper(A,N,Acc) when A == -1 ->
	case is_even(N) of
		{ok,true} -> {?JACOBI_BAD_ARG,N};
		{ok,false} -> 
			case N rem 4 of
		1 -> 1 * Acc;
		3 -> -1 * Acc;
		_ -> {bad_argument,N}
	end  
		
	end;

jacobi_helper(A,N,Acc) when N rem 2 == 0 ->
	{?JACOBI_BAD_ARG,"N cannot be even"};


jacobi_helper(A,N,Acc) when A == 2  ->
	debug("Current value for Acc = ~p for j(~p,~p)~n",[Acc,A,N]),
		case is_even(N) of 
		{ok,true} ->{?JACOBI_BAD_ARG,"N cannot be even"} ;
		{ok,false} ->
			case (N rem 8) of
				1 -> 1 * Acc;
				7 ->  1 * Acc;
				3 -> -1 * Acc;
				5 -> -1 * Acc;
			    _ -> {?JACOBI_BAD_ARG,"jacobi property for A",[A,N]}
			end
	end;
	

jacobi_helper(A,N,Acc) when is_integer(A),is_integer(N) ->
	debug("Current value for Acc = ~p for j(~p,~p)~n",[Acc,A,N]),
	case is_even(A) of
		{ok,true} ->
			{OddNum,Exp} = div_till_odd(A),
			debug("(jacobi(~p/~p))^~p , jacobi(~p,~p)~n",[2,N,Exp,OddNum,N]),
			Twores = jacobi_helper(2,N,Acc),
			jacobi_helper(OddNum,N,round(math:pow(Twores,Exp))*Acc);
		{ok,false} ->
 			case A > N  of
				true ->
					Decomp = A rem N,
					io:format("Decomp = ~p~n",[Decomp]),
  					debug("(Decomposed odd =jacobi(~p/~p)~n",[Decomp,N]),
 					jacobi_helper(Decomp,N,Acc);					
				false ->
					Flip_res = jacobi_flip(A,N),
 					debug("flip ==> (~p)jacobi(~p/~p)~n",[Flip_res,N,A]),
  					jacobi_helper(N,A,Flip_res* Acc) 
			end
	end;
 

jacobi_helper(_,_,Acc)  -> Acc.


	

%%Utility code
%% TODO: TO BE IN SEPERATE MODULE
-spec verify(Property,Args) -> Bool when
		  Property :: atom(),
		  Args :: [term()],
		  Bool :: boolean().

verify(jacobi_prop,Args) when length(Args) > 1 ->
	JProp = #jacobi_prop{},
	{ok,Numer,N,AProduct,NProduct} = process_args(jacobi,Args),
	 denom_check(jacobi,N,{(fun(A) -> is_even(A) end),{ok,true},{ok,false}},
				(fun(A) -> is_composite(A) end)) ,
		result_check(jacobi,AProduct,NProduct),
	prop_check(jacobi,1,Numer,N),
	prop_check(jacobi,2,Numer,NProduct) ,
	prop_check(jacobi,3,AProduct,N),
	prop_check(jacobi,4,[5,8],33),
	prop_check(jacobi,5,5,33),
	prop_check(jacobi,6,-1,33),
	prop_check(jacobi,7,2,33),
	prop_check(jacobi,8,5,33);

%%numerator and prime check (Definition).
verify(jacobi_prop,Args) when length(Args) =:= 1 ->
	?NOT_YET_IMPLEMENTED;
verify(_,Args) -> 
	{bad_argument,Args}.

-spec prop_check(Jacobi,Num,A,N) -> {Ok,VerifyAtom} when
		  Jacobi 		:: atom(),
		  Num			:: integer(),
		  A		 		:: [integer()] | integer(),
		  N 				:: integer() | [integer()],
		  Ok				:: atom(),
		  VerifyAtom 	:: atom().
prop_check(jacobi,Num,A,N) when Num =:= 1 -> 
	case is_list(A) andalso length(A) > 1 of
		true ->
			Res = lists:foldl(fun(Elem,Acc) ->
				jacobi(Elem,case is_list(N) of true->multiply(list,N); false -> N end) * Acc
				end, 1,A),
			A2 = multiply(list,A),
			Res2 = jacobi(A2,case is_list(N) of true->multiply(list,N); false -> N end),
			debug("Jacobi property (ii) jacobi(~p/~p) == ",
				  [A2,multiply(list,N)]),
				%[ io:format("~p/~p  ",[X,multiply(list,N)]) || X<- A],
				%io:format(" = ~p~n",[(Res == Res2)]);
			print_multiple_numerator_vs_single_denom(A,N,"*",length(A),1),
			io:format(" = ~p~n",[(Res == Res2)]);
        false ->
           debug("Jacobi Property (ii) Test Requirement : Numerator must be list~n",[])
        end;

prop_check(jacobi,Num,AProduct,N) when Num =:= 2 -> 
	io:format("AProduct : ~p~n",[AProduct]),
	Fun = fun(A,B) -> jacobi(A,B) end,
	Mul_denom_arg_res = apply_across_args(accum_mul,Fun,AProduct,N),
    case Mul_denom_arg_res =:= jacobi(multiply(list,AProduct),N) of
		true -> {ok,verify_passed_prop2};
		false ->{ok,verify_failed_prop,concat_str_to_integer("property ",Num," ")}
    end;

prop_check(jacobi,Num,A,N) when Num =:= 3  ->		
        case is_list(N) andalso length(N) > 1 of
			true ->
			Res = lists:foldl(fun(Elem,Acc) ->
				jacobi(A,case is_list(N) of true->multiply(list,N); false -> N end) * Acc
				end, 1,N),
			NProduct = multiply(list,N),
			Res2 = jacobi(A,NProduct),
			debug("Jacobi property (iii) jacobi(~p/~p) == ",
				  [A,multiply(list,N)]),
			print_multiple_denumerator_vs_single_numer(N,A,"*",length(N),1),
			io:format(" = ~p~n",[(Res == Res2)]);
        false ->
           debug("Jacobi Property (iii) Test Requirement : Denumerator must be list~n",[])
        end;

prop_check(jacobi,Num,[A,B],N) when Num =:= 4  ->
	case A =:= B rem N of
		true -> 
				 debug("Jacobi Property(iv) jacobi(~p/~p) == jacobi(~p/~p) = ~p~n",
							  [A,N,B,N,(jacobi(A,N) =:= jacobi(B,N) )]);
		false ->debug(?UNDEFINED_PROPERTY_CONDITION,[Num]),
				{ok,verify_failed,concat_str_to_integer("property ",Num," ")}
    end;

prop_check(jacobi,Num,A,N) when Num =:= 5 ->
	debug("Jacobi Property(v) jacobi(~p/~p) == 1 ? = ~p~n",
							  [A,N,(jacobi(A,N) =:= 1 )]),
	{ok,(case jacobi(A,N) =:= 1 of true -> ?V_PASS; false -> ?V_FAIL end),concat_str_to_integer("property ",Num," ")};
prop_check(jacobi,Num,A,N) when Num =:= 6 ->
		J = jacobi(A,N),
		debug("Jacobi Property(vi) jacobi(~p/~p) == -1 or 1 ? = ~p~n ...performing further test for this property~n",
							  [A,N,(case (J =:= -1) or (J =:= 1) of true -> true; false -> false end )]),
	quick_test(jacobi_property,0,-1,0),
	{ok,(case J =:= 1 of true -> ?V_PASS; false -> ?V_FAIL end),
	 concat_str_to_integer("property ",Num," ")};

prop_check(jacobi,Num,A,N) when Num =:= 7 ->
		J = jacobi(A,N),
		debug("Jacobi Property(vii) jacobi(~p/~p) == -1 or 1 ? = ~p~n ...performing further test for this property~n",
							  [A,N,(case (J =:= -1) or (J =:= 1) of true -> true; false -> false end )]),
	quick_test(jacobi_property,7,-1,50),
	{ok,(case J =:= 1 of true -> ?V_PASS; false -> ?V_FAIL end),
	 concat_str_to_integer("property ",Num," ")};

prop_check(jacobi,Num,A,N) when Num =:= 8 ->
	J = jacobi(A,N),
	J2 = jacobi(N,A),
	debug("Jacobi Property(viii) jacobi(~p/~p) == jacobi(~p/~p) ? = ~p~n ...performing further test for this property~n",
			[A,N,N,A, 
					case (J2 == J) of true -> true; 
						false -> 
							debug("Performing further test...~n
                                Checking Congruence to  3 mod 4..~n",[]),
						    case (A rem 4 == 3) andalso (N rem 4 == 3) of 
							  true -> debug("Jacobi(~p/~p) and jacobi(~p/~n) ==? ~p~n",
								[A,N,N,A,
								  case (jacobi(A,N)) =:= -(jacobi(N,A))  of
									  true -> true; 
									  false-> error(?V_FAIL ++", 3 mod 4 congurnce failed for property 8 ")
								  end]) ;
							  false -> error("Bad Jacobi object")
							end
					end ]),
	{ok, ?V_PASS,
	 concat_str_to_integer("property ",Num," ")};
	



prop_check(jacobi,_,Num,_) -> {error,unknown_property,Num}.

-spec process_args(Atom,Args) -> {Ok,Numer,N,AProduct,NProduct} when
		  Atom  	 	:: atom(),
		  Args  		:: [integer()],
	      Ok    		:: atom(),
		  Numer 		:: [integer()],
	      N     		:: [integer()],
	      AProduct 	:: integer(),
		  NProduct 	:: integer().
process_args(jacobi,Args) ->
	{Numer,[N]} = lists:split(length(Args)-1,Args),
	NProduct = lists:foldl(fun(Ele,Acc) -> Ele * Acc end, 1,N),
	AProduct = lists:foldl(fun(Ele,Acc) -> Ele * Acc end, 1,Numer),
	{ok,Numer,N,AProduct,NProduct}.

denom_check(jacobi,N,{Condition1,MatchClause1,MatchClause2},Condition2) ->
	io:format("denom_check called (args: N=~p)~n",[N]),	
	lists:filter(fun(Elem) -> 
						 case Condition1(Elem)  of
							 MatchClause1  ->error("Bad argument denom not odd"); % {jacobi_error,"Bad argument denom not odd"};								 
						      MatchClause2 -> 
								  case Condition2(Elem) of
									  true -> true;
									  false -> error("Bad argument denom should not be prime") %{jacobi_error,"Bad argument denom should not be prime"}
								  end								  
						 end
				 end,N).

result_check(jacobi,AProduct,NProduct) ->	
	case jacobi(AProduct,NProduct) of
		0 -> 
			debug("result_check(AProduct=~p,NProduct=~p) return:~p~n",[AProduct,NProduct,{ok,true,0}]),
			{ok,true};
		1 ->debug("result_check(AProduct=~p,NProduct=~p) return:~p~n",[AProduct,NProduct,{ok,true,1}]), 
			{ok,true};
		-1 ->debug("result_check(AProduct=~p,NProduct=~p) return:~p~n",[AProduct,NProduct,{ok,true,-1}]), 
			{ok,true};
		Other ->debug("result_check(AProduct=~p,NProduct=~p) return:~p~n",[AProduct,NProduct,{error,illegal_output,Other}]), 
			{error,illegal_output}
    end.

 -spec apply_across_args(Accum,Function,Args,N) -> Result when
 		  Accum 		:: atom(),
 		  Function 	:: function(),
 		  Args 		:: [integer()],
 		  N     		:: [integer()],
 		  Result 	:: integer().

apply_across_args(accum_mul,Function,Args,N) when (length(Args) > 1)  ->
	lists:foldl(fun(Elem,Acc) ->
				io:format("Arg ~p N : ~p~n",[Elem,N]),
				Function(Elem,N) * (Acc)
				end, 1,Args);

 apply_across_args(accum_mul,Function,Args,N) when (length(Args) == 1) ->
 	Function(hd(Args),N);
 apply_across_args(accum_mul,_,_,N)  ->
	io:format("N ~p~n",[2*N]),
	shouldnt_get_here.


%%Utility code
%% TODO: TO BE IN SEPERATE MODULE
-spec convert2(BaseType,Number) -> List when
		  BaseType :: atom(),
		  Number :: integer(),
		  List :: list().
convert2(base2,Number) ->
	lists:reverse([N - $0 || N <-  integer_to_list(Number,2)]).

-spec numeric_rep(Number,Base) -> [{Base,Exp,Rep}] when
		  Number :: [integer()],
		  Base 	 :: integer(),
		  Exp 	 :: integer(),
		  Rep    :: integer().
numeric_rep(Number,Base) ->
	numeric_rep(Number,Base,[],0).

numeric_rep([H|T],Base,Acc,Counter) ->
	numeric_rep(T,Base,[{Base,Counter,H - $0} | Acc ],Counter+1);
numeric_rep([],Base,Acc,Counter) -> lists:reverse(Acc).
	
-spec is_even(Number) -> {Atom,Boolean} when
		  Number 	:: integer(),
		  Atom	 	:: atom(),
		  Boolean	:: boolean().
is_even(Number) when is_integer(Number) , Number >= 0 ->
    {ok,(Number band 1 ) == 0};
is_even(Number) -> {bad_argument,Number}.
-spec is_odd(Number) -> {Atom,Boolean} when
		  Number 	:: integer(),
		  Atom	 	:: atom(),
		  Boolean	:: boolean().

is_odd(Number) when is_integer(Number) , Number >= 0 ->
	{ok,(Number band 1) == 1};
is_odd(Number) -> {bad_argument,Number}.

-spec multiply(Atom,List) -> Integer when
	Atom		:: atom(),
	List		:: list(),
	Integer	:: integer().
multiply(list,List) ->
	lists:foldl(fun(A,Acc) -> 
					Acc * A		
				end,1,List).

-spec debug(String,Args) -> NoReturn when
		  String		:: string(),
		  Args		:: [term()],
		  NoReturn  :: no_return().
debug(String,Args) ->
  io:format(String,Args).


-spec print_multiple_numerator_vs_single_denom(Numer,Denom,Op,L) -> NoReturn when
	Numer		:: [integer()],
	Denom		:: integer(),
	Op			:: term(),
	L			:: integer(),
	NoReturn		:: no_return().
print_multiple_numerator_vs_single_denom(Numer,Denom,Op,L)  ->
	print_multiple_numerator_vs_single_denom(Numer,Denom,Op,L,0).

print_multiple_numerator_vs_single_denom([H|T],Denom,Op,NumerL,Counter) ->
	case Counter == NumerL of 
		true -> io:format("jacobi(~p/~p)  ",[H,multiply(list,Denom)]),
				print_multiple_numerator_vs_single_denom(T,Denom,Op,NumerL,Counter+1);
		false -> io:format("jacobi(~p/~p)~p",[H,multiply(list,Denom),list_to_atom(Op)]),
				print_multiple_numerator_vs_single_denom(T,Denom,Op,NumerL,Counter+1)
		end;

print_multiple_numerator_vs_single_denom([],Denom,Op,NumerL,Counter) ->
	io:format(" ").

-spec print_multiple_denumerator_vs_single_numer(Numer,Denom,Op,L) -> NoReturn when
	Numer		:: integer(),
	Denom		:: [integer()],
	Op			:: term(),
	L			:: integer(),
	NoReturn		:: no_return().
print_multiple_denumerator_vs_single_numer(Denom,Numer,Op,L)  ->
	print_multiple_denumerator_vs_single_numer(Denom,Numer,Op,L,0).

print_multiple_denumerator_vs_single_numer([H|T],Numer,Op,DenomL,Counter) ->
	case Counter == DenomL of 
		true -> io:format("jacobi(~p/~p)  ",[Numer,H]),
				print_multiple_denumerator_vs_single_numer(T,Numer,Op,DenomL,Counter+1);
		false -> io:format("jacobi(~p/~p)~p",[Numer,H,list_to_atom(Op)]),
				print_multiple_denumerator_vs_single_numer(T,Numer,Op,DenomL,Counter+1)
		end;

print_multiple_denumerator_vs_single_numer([],Numer,Op,DenomL,Counter) ->
	io:format(" ").

concat_str_to_integer(String,Integer,Sep) ->
	String++ Sep ++  integer_to_list(Integer).

quick_test(jacobi_property,Num,A,NSeq) when Num =:= 6 ->  
	%[io:format("~p~n",[{integer_to_list(A) ++ "," ++ integer_to_list(X) ++ " = " ++integer_to_list(round(math:pow(A,(X-1) div 2)))} || X <- lists:seq(3,NSeq) , X rem 2 == 1, crypto_thots:is_composite(X)])],
	[{A,X,(round(math:pow(A,(X-1) div 2)))} || X <- lists:seq(3,NSeq) , X rem 2 == 1, crypto_thots:is_composite(X)];


quick_test(jacobi_property,Num,A,NSeq) when Num =:= 0 ->
	Res =  quick_test(jacobi_property,6,-1,100) ,
	[ {A,X,case (X rem 4 == 1) of true -> true; false -> error(?V_FAIL++ " N Mod 4,Property 6") end}  ||  {A,X,R} <- Res , R =:= 1 ],
	debug("Jacobi Property(vi) jacobi(-1/N) == 1 MOD 4 ? = true ~n ...performing further test for this property~n",[]),
	%[A,N,(case (J =:= -1) or (J =:= 1) of true -> true; false -> false end )]),
	[ {A,X,case (X rem 4 == 3) of 
			   true -> true; 
			   false -> error(?V_FAIL++ " N Mod 4,Property 6") 
		   end}  ||  {A,X,R} <- Res , R =:= -1 ],
		  debug("Jacobi Property(vi) jacobi(-1/N) == 3 MOD 4 ? = true ~n",[]);


quick_test(jacobi_property,Num,A,NSeq=50) when Num =:= 7 ->
	Res = [{X,round(math:pow(-1,round(math:pow(X,2) -1) div 8))} || X <- lists:seq(3,NSeq) , X rem 2 == 1, crypto_thots:is_composite(X)],
		[ {A,X,case (X rem 8 == 1) or (X rem 8 == 7) of true -> true; false -> error(?V_FAIL++ " N Mod 4,Property 6") end}  ||  {A,X,R} <- Res , R =:= 1 ],
	debug("Jacobi Property(vi) jacobi(2/N) == 1 or 7 MOD 8  ? = true ~n ...performing further test for this property~n",[]),
	%[A,N,(case (J =:= -1) or (J =:= 1) of true -> true; false -> false end )]),
	[ {A,X,case (X rem 8 == 3) or (X rem 8 == 5) of 
			   true -> true; 
			   false -> error(?V_FAIL++ " N Mod 8,Property 7") 
		   end}  ||  {A,X,R} <- Res , R =:= -1 ],
		  debug("Jacobi Property(vii) jacobi(2/N) == 3 or 5 MOD 8 ? = true ~n",[]);

quick_test(jacobi_property,_,_,_) ->
	not_yet_implemented.


-spec div_till_odd(BigNum) -> {OddNum,Exp} when 
		  BigNum :: integer(),
		  OddNum :: integer(),
		  Exp    :: integer(). 
div_till_odd(BigNum) ->
	Div_till_odd = fun The_fun(BigNum,Acc) when BigNum rem 2 == 0 ->  The_fun(BigNum bsr 1,Acc+1)  ; The_fun(BigNum,Acc) -> {BigNum,Acc} end,
	Div_till_odd(BigNum,0).
		  
 
-spec jacobi_flip(M,N) -> Res when
		  M 		:: integer(),
		  N 		:: integer(),
	      Res	:: jacobi_result().
jacobi_flip(M,N) ->
	%debug("flip => jacobi(~p/~p)~n",[N,M]),
	case (M rem 4) == 3 andalso (N rem 4) == 3 of
		true -> -1;
		false -> 1
	end.

pow_bin(X,N) -> 
	pb(X,N,1).
pb(X,N,Acc) when (N rem 2) =:= 0 -> 
	pb(X*X, N div 2 ,Acc) ; 
pb(X,N,Acc) -> 
	NewAcc = Acc * X,
	case  N div 2 of
		0 -> NewAcc; 
		_ -> pb(X*X,N div 2,Acc * X) 
	end.






				
	
		
	
