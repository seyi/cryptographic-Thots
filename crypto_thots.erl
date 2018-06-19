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
		 modular_exponent/4,is_quad_res/2,is_even/1,is_odd/1,jacobi/2,
		 verify/2]
		).
-define(NOT_YET_IMPLEMENTED,not_yet_implemented).
-define(V_FAIL,verify_failure).
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

%%    Mod = round(math:pow(10,9) + 7),
%%     round( 
%%           ((
%%             (round(X rem Mod)
%%                * round(Y rem Mod) 
%%              ) 
%%                *Mod
%%            ) 
%%              /  (gcd(euclid,(X rem Mod),(Y rem Mod))))  / Mod
%%          );
    
%%RSA-576
%%  1881988129206079638386972394616504398071635633794173827007
%%    6335642298885971523466548531906060650474304531738801130339
%%    6716199692321205734031879550656996221305168759307650257059


%%RSA-155
%%1094173864157052742180970732204035761200373294544920599091 3842131476349984288934784717997257891267332497625752899781 833797076537244027146743531593354333897.

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

-spec convert2(BaseType,Number) -> List when
		  BaseType :: atom(),
		  Number :: integer(),
		  List :: list().
convert2(base2,Number) ->
	lists:reverse([N - $0 || N <-  integer_to_list(Number,2)]).

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


-spec is_quad_res(N,P) -> Boolean
     when 
		  N :: integer(),
		  P :: odd_prime(),
		  Boolean :: boolean().
is_quad_res(N,P) ->
	case Res = (round(math:pow(N,(P-1) div 2)) rem P) of
		1 -> {true, Res};
		_ -> {false , Res}
     end.

%%Jacobi Properties
-spec jacobi(Args,N) -> Number when
		Args ::[number()] | integer(),
		N    :: [odd_number()] | odd_number(),
		Number :: jacobi_result.
 jacobi(A,N) when is_integer(A),is_integer(N) -> 
	2;

jacobi(A,N) when is_list(N) andalso is_integer(A) ->
		3;
jacobi(Args,N) when is_list(Args) andalso is_integer(N) -> 
		4;
jacobi(Args,N) when is_list(Args) andalso is_list(N) -> 
 5.	

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
	prop_check(jacobi,4,[5,8],33);

%%numerator and prime check (Definition).
verify(jacobi_prop,Args) when length(Args) =:= 1 ->
	?NOT_YET_IMPLEMENTED;
verify(_,Args) -> 
	{bad_argument,Args}.

-spec prop_check(Jacobi,Num,A,N) -> {Ok,VerifyAtom} when
		  Jacobi 		:: atom(),
		  Num			:: integer(),
		  A		 		:: [integer()],
		  N 				:: integer() | [integer()],
		  Ok				:: atom(),
		  VerifyAtom 	:: atom().
prop_check(jacobi,Num,A,N) when Num =:= 1 -> 
	case is_list(A) andalso length(A) > 1 of
		true ->
			Res = lists:foldl(fun(Elem,Acc) ->
				jacobi(Elem,N) * Acc
				end, 1,A),
			A2 = multiply(list,A),
			Res2 = jacobi(A2,N),
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
		false ->{ok,verify_failed_prop2}
    end;

prop_check(jacobi,Num,A,N) when Num =:= 3  ->
		
        case is_list(N) andalso length(N) > 1 of
			true ->
			Res = lists:foldl(fun(Elem,Acc) ->
				jacobi(A,N) * Acc
				end, 1,N),
			NProduct = multiply(list,N),
			Res2 = jacobi(A,NProduct),
			debug("Jacobi property (iii) jacobi(~p/~p) == ",
				  [A,multiply(list,N)]),
				%[ io:format("~p/~p  ",[X,multiply(list,N)]) || X<- A],
				%io:format(" = ~p~n",[(Res == Res2)]);
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
		false ->debug(?UNDEFINED_PROPERTY_CONDITION,[Num])
    end;

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
is_even(Number) when is_integer(Number) , Number >= 0 ->
    {ok,(Number band 1 ) == 0};
is_even(Number) -> {bad_argument,Number}.
is_odd(Number) when is_integer(Number) , Number >= 0 ->
	{ok,(Number band 1) == 1};
is_odd(Number) -> {bad_argument,Number}.

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



				
	
		
	
