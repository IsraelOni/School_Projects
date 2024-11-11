%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prolog Lab 1
% Nov. 12 and 14, 2022
% by R. Heise
%
% Simple examples of coding in Prolog.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get most fun Reading Week activity
% and favorite foods
%

funActivity(rosanna, walking_in_snow).

% Monday Lab
funActivity(aarushi, scheming).
%funActivity(abdullah, _).
funActivity(anjola, watchingTV).
funActivity(danielKi, playingVideoGames).
funActivity(danielKr, watchingFootball).
funActivity(harry, gaming).
funActivity(israel, kurtzpell).
funActivity(jamesM, playingVideoGames).
funActivity(jamesR, playingVideoGames).
funActivity(jordan, playingVideoGames).
funActivity(priscilla, sleeping).
funActivity(sandile, fifa).
funActivity(thai, coding).
%funActivity(tina, _).
funActivity(warren, boardGames).
funActivity(zach, gambling).
% Wednesday Lab
%funActivity(boma, _).
funActivity(brayden, sleeping).
%funActivity(derin, _).
funActivity(fei, basketball).
funActivity(keelan, hangingOut).
funActivity(runqi, arknights).
funActivity(yixiao, teaching).

favFood(boma, blueberries).
favFood(aarushi, pizza).
%favFood(abdullah, _).
favFood(anjola, sushi).
favFood(danielKi, sushi).
favFood(danielKr, sushi).
favFood(harry, nandos).
favFood(israel, bananas).
favFood(jamesM, koreanBBQ).
favFood(jamesR, pasta).
favFood(jordan, sanwiches).
favFood(priscilla, kiwis).
favFood(sandile, friedRice).
favFood(thai, carbonara).
%favFood(tina, _).
favFood(warren, lasagna).
favFood(zach, padThai).




%favFood(boma, _).
favFood(brayden, tacos).
%favFood(derin, _).
favFood(fei, dumplings).
favFood(keelan, curry).
favFood(runqi, bananas).
favFood(yixiao, chicken).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write predicates for combinations
%
studentsWhoGetA(Name) :- funActivity(Name, coding).
studentsWhoGetA(Name) :- funActivity(Name, scheming).
studentsWhoGetA(Name) :- favFood(Name, blueberries).

studentsWhoFail(Name) :- funActivity(Name, playingVideoGames),
    favFood(Name, sushi).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lists are used for storing multiple items
%
% Make a list of foods. Write a predicate isFood that is true if its
% parameter is a food.
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% allRs(List) will be true if the list is completely made of 'r's
%
allRs([]).
allRs([r | Tail]) :- allRs(Tail).

fixedNumRs(0, []).
fixedNumRs(Size, [r | Rest]) :- Size > 0,
    NewSize is Size - 1,
    fixedNumRs(NewSize, Rest).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% myMember(Element, List) is equivalent to Prolog's member predicate
% Returns true if the first argument in inside the 2nd argument
%
myMember(Element, [Element | _]).
myMember(Element, [SomethingElse | Rest]) :-
    Element \== SomethingElse,
    myMember(Element, Rest).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% myAppend(List1, List2, List3)  equivalent to Prolog's append predicate
% returns true if the third argument is the first 2 arguments
% put together
%
%myAppend([], [], []).
%myAppend(List1, [], List1).
myAppend([], List2, List2).
myAppend([FirstOfList1 | RestOfList1], List2,
         [FirstOfList1 | More]) :-
    myAppend(RestOfList1, List2, More).

%frontAndLast(Front, LastElement, AList)
frontAndLast([], Element, [Element]).
frontAndLast([FirstElement | RestFront], Element,
             [FirstElement | Rest]) :- Rest \== [],
             frontAndLast(RestFront, Element, Rest).

frontAndLast(Backshots , Lips, [1,2,3,4,5]), write(Lips).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% myReverse(List1, List2) equivalent to Prolog's reverse
% returns true if 2nd argument is the first argument reversed
%
%myReverse([OneThing], [OneThing]).
myReverse([], []).
myReverse([First | Rest], Answer) :- myReverse(Rest, RestRev),
    myAppend(RestRev, [First], Answer).


% tail recursive?


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% max(Item, List) is true if Item is the biggest value in the list
% of integers
%
%max(OneThing, [OneThing]).
max(Answer, [First | Rest]) :- maxHelper(First, Rest, Answer).

%Assume the when maxHelper is first called, List contains
%at least one item.
%maxHelper(CurrentMax, List, FinalAnswer)
maxHelper(CurrentMax, [], CurrentMax).
maxHelper(CurrentMax, [First | Rest], FinalMax) :-
    First > CurrentMax,
    maxHelper(First, Rest, FinalMax).
maxHelper(CurrentMax, [First | Rest], FinalMax) :-
    First =< CurrentMax,
    maxHelper(CurrentMax, Rest, FinalMax).
