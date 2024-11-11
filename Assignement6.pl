
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author : Israel Oni
% Date : December 05, 2022
%
% Description: Unbeknownst to most of us, Santa still requires money
% to obtain the raw resources to make all the toys for Christmas. Aside
% from being skilled toy-makers, Santa’s elves are also incredible
% academics and are always working on new textbooks, except during the
% very busy Autumn preceding the even busier Christmas season. Over the
% past number of years, each of Santa’s elves (whose names are Alabaster
% Snowball, Bushy Evergreen, Missie Toe, Pepper Minstix, Shinny Upatree,
% Sugarplum Mary, and Wunorse Openslae) has published a textbook in
% their area of expertise. These textbooks include Algorithms for
% Parallel Reindeer, Data Structures for Best Cookies, Scheme for
% Smarties, Ethics of Present Delivery, Discrete Math for Toy-Making,
% Calculus Most Irrelevant, and North Pole Algebra. These textbooks are
% sold, for prices of $250, $300, $350, $400, $450, $500, and $550
% depending on the textbook, to students to aid in the learning of
% critical subject areas at the University of Alberta. The elves
% completed the textbooks during various summers in the years 2012,
% 2014, 2015, 2017, 2018, 2019, and 2020. University students have found
% the textbooks to be of excellent quality, and the price to be most
% reasonable.
%
% Summary: This file contains predicates that will act as facts that
% will tell prolog information that it will use to find a unique answer.
% There is also a predicate that will print the answer in a lined up
% 7x7 format. Lastly there will be a predicate that will contain a list
% of lists where all the elfs are instantiated and each list will
% contain 3 blanks with one unique elf.
%
%
%
%
%
%
%
%
%
%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% print2D(Grid) Prints out a list of lists (Grid) to standard output so
% it looks 2 dimensional.
%
print2D([]) :- nl.
print2D([FirstRow | Rest]) :- printRow(FirstRow),
    print2D(Rest).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% printRow(ARow)
% Helper to print2D, used to print a single row of a 2D board.
% In this case it will print from the database of elfs,textbooks,years
% and prices output it in a string
printRow([]) :- nl.
printRow([Item | Rest]) :- Item == elf1, writef("%20l", ['Alabaster SnowBall']), printRow(Rest).
printRow([Item | Rest]) :- Item  == elf2, writef("%20l", ['Bushy EverGreen']),printRow(Rest).
printRow([Item | Rest]) :- Item  == elf3, writef("%20l", ['Missie Toe']), printRow(Rest).
printRow([Item | Rest]) :- Item  == elf4, writef("%20l", ['Pepper Minstix']) ,printRow(Rest).
printRow([Item | Rest]) :- Item  == elf5, writef("%20l", ['Shinny Upatree']),printRow(Rest).
printRow([Item | Rest]) :- Item  == elf6, writef("%20l", ['Sugarplum Mary']),printRow(Rest).
printRow([Item | Rest]) :- Item  == elf7, writef("%20l", ['Wunorse Openslae']),printRow(Rest).




printRow([Item | Rest]) :- Item  == textbook1, writef("%35l", ['Algorithms for Parallell Reindeer']),printRow(Rest).
printRow([Item | Rest]) :- Item  == textbook2, writef("%35l", ['Data Structures for Best Cookies']),printRow(Rest).
printRow([Item | Rest]) :- Item  == textbook3, writef("%35l", ['Scheme for Smarties']),printRow(Rest).
printRow([Item | Rest]) :- Item  == textbook4, writef("%35l", ['Ethics of Present Delivery']),printRow(Rest).
printRow([Item | Rest]) :- Item  == textbook5, writef("%35l", ['Discrete Math for Toy Making']),printRow(Rest).
printRow([Item | Rest]) :- Item  == textbook6, writef("%35l", ['Calculus Most Irrelevant']),printRow(Rest).
printRow([Item | Rest]) :- Item  == textbook7, writef("%35l", ['North Pole Algebra']),printRow(Rest).




printRow([Item | Rest]) :- Item  == 2012, writef("%5c", ['2012']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 2014, writef("%5c", ['2014']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 2015, writef("%5c", ['2015']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 2017, writef("%5c", ['2017']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 2018, writef("%5c", ['2018']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 2019, writef("%5c", ['2019']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 2020, writef("%5c", ['2020']),printRow(Rest).

printRow([Item | Rest]) :- Item  == 250, writef("%5c", ['$250.00']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 300, writef("%5c", ['$300.00']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 350, writef("%5c", ['$350.00']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 400, writef("%5c", ['$400.00']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 450, writef("%5c", ['$450.00']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 500, writef("%5c", ['$500.00']),printRow(Rest).
printRow([Item | Rest]) :- Item  == 550, writef("%5c", ['$550.00']),printRow(Rest).




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DataBase for all the textbooks,Prices,Years for the puzzle
%
%
%
%

textbooks([textbook1,textbook2,textbook3,textbook4,textbook5,textbook6,textbook7]).
years([2012, 2014, 2015, 2017, 2018, 2019, 2020]).
prices([250, 300, 350, 400, 450, 500, 550]).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this predicate  contains a list
% of lists where all the elfs are instantiated and each list will
% contain 3 blanks with one unique elf.
% The blanks will be filled in by the predicates below
%
%
%
%
solution([
          [elf1, _, _,_],
          [elf2, _, _,_],
          [elf3, _, _,_],
          [elf4, _, _,_],
          [elf5, _, _,_],
          [elf6, _, _,_],
          [elf7, _, _,_]
         ]).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This predicate will contain other predicate that will solve the puzzle
% in the predicate solution.
%
%
%
solvePuzzle :-
    solution(Answer),
    textbook2018ByMissie(Answer),
    book2020OrCost550WrittenByMissieOrSugarplum(Answer),
    dataStructureand2017WrittenByPepperOrCost400(Answer),
    ethicCostLessThanTextbook2017(Answer),
    ethicCosts150LessThanNorth(Answer),
    northCost50LessThanTextbook2012(Answer),
    textbookPepperCost100LessThanTextbook2014(Answer),
    textbook2019Cost100MoreScheme(Answer),
    uniquenessForAllThePresentedTitles(Answer),
    textbook2020Cost50LessThanShinnyTextbook(Answer),
    alabasterNotAuthorDiscreteNorTextbookPrice400(Answer),
    noDuplicate(Answer),
    print2D(Answer).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   North Pole Algebra costs $50.00 less than the textbook published in 2012.
%
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate
%
%
northCost50LessThanTextbook2012(Answer) :-  member([_,textbook7,Textbook7Year,Textbook7Price],Answer),
                 member([_,Textbook2012,2012,Prices2012],Answer),
                 Textbook7Year \== 2012,
                 Textbook2012 \== textbook7,
                 prices(PossiblePrices),
                 member(Prices2012, PossiblePrices),
                 member(Textbook7Price, PossiblePrices),
                 Prices2012 is Textbook7Price + 50.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ethics of Present Delivery costs $150.00 less than North Pole Algebra.
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate


ethicCosts150LessThanNorth(Answer) :- member([_,textbook4,Textbook4Year,Textbook4Price],Answer),
    member([_,textbook7,Textbook7Year,Textbook7Price], Answer),
    Textbook4Year \== Textbook7Year,
    prices(PossiblePrices),
    member(Textbook4Price,PossiblePrices),
    member(Textbook7Price,PossiblePrices),
    Textbook7Price is Textbook4Price + 150.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The $500.00 title, the textbook written by Pepper Minstix, the
% textbook written by Shinny Upatree, the textbook that was published in
% 2018, the textbook written by Wunorse Openslae and Calculus Most
% Irrelevant are all unique textbooks.
%
% This predicate makes sure all the titles are unique by unbinding the
% titles that appear in the same list
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate

%


uniquenessForAllThePresentedTitles(Answer) :-prices(PossiblePrices),
                textbooks(PossibleTextbook),
                years(PossibleYears),
                member([Author1,Textbook1,Year1,500],Answer),
                member([elf5,Textbook2,Year2,Price1],Answer),
                member([elf7,Textbook3,Year3,Price2],Answer),
                member([Author2,textbook6,Year4,Price3],Answer),
                member([Author3,Textbook4,2018,Price4],Answer),
                member([elf4,Textbook5,Year5,Price5],Answer),

                not(member(Author1,[elf5,elf6,elf4,Author2,Author3])),
                not(member(Author2,[elf5,elf6,elf4,Author1,Author3])),
                not(member(Author3,[elf5,elf6,elf4,Author1,Author2])),


                member(Year1,PossibleYears),
                member(Year2,PossibleYears),
                member(Year3,PossibleYears),
                member(Year4,PossibleYears),
                member(Year5,PossibleYears),

                  Year1 \== Year2,
                  Year1 \== Year3,
                  Year1 \== Year4,
                  Year1 \== Year5,

                  Year2 \== Year3,
                  Year2 \== Year4,
                  Year2 \== Year5,

                  Year3 \== Year4,
                  Year3 \== Year5,

                  Year4 \== Year5,


                  Year1 \== 2018,
                  Year2 \== 2018,
                  Year3 \== 2018,
                  Year4 \== 2018,
                  Year5 \== 2018,


                  member(Textbook1,PossibleTextbook),
                  member(Textbook2,PossibleTextbook),
                  member(Textbook3,PossibleTextbook),
                  member(Textbook4,PossibleTextbook),
                  member(Textbook5,PossibleTextbook),

                  Textbook1 \== Textbook2,
                  Textbook1 \== Textbook3,
                  Textbook1 \== Textbook4,
                  Textbook1 \== Textbook5,

                  Textbook2 \== Textbook3,
                  Textbook2 \== Textbook4,
                  Textbook1 \== Textbook5,

                  Textbook3 \== Textbook4,
                  Textbook3 \== Textbook5,


                  Textbook4 \== Textbook5,

                  Textbook1 \== textbook6,
                  Textbook2 \== textbook6,
                  Textbook3 \== textbook6,
                  Textbook4 \== textbook6,
                  Textbook5 \== textbook6,

                member(Price1,PossiblePrices),
                member(Price2,PossiblePrices),
                member(Price3,PossiblePrices),
                member(Price4,PossiblePrices),
                member(Price5,PossiblePrices),

                 Price1 \== Price2,
                 Price1 \== Price3,
                 Price1 \== Price4,
                 Price1 \== Price5,


                 Price2 \== Price3,
                 Price2 \== Price4,
                 Price2 \== Price5,

                 Price3 \== Price4,
                 Price3 \== Price5,

                 Price4 \== Price5,

                  Price1 \== 500,
                  Price2 \== 500,
                  Price3 \== 500,
                  Price4 \== 500,
                  Price5 \== 500.














%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% The textbook that was published in 2020 and the one that cost $550.00, were written
% by either Missie Toe or Sugarplum Mary.
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate


book2020OrCost550WrittenByMissieOrSugarplum(Answer) :- member([elf6,Elf6Textbook,2020, Elf6Price], Answer),
                 member([elf3,Elf3Textbook,Elf3Year,550],Answer),
                 Elf6Textbook \== Elf3Textbook,
                 Elf3Year \== 2020,
                 Elf6Price \== 550.
book2020OrCost550WrittenByMissieOrSugarplum(Answer) :- member([elf3,Elf3Textbook,2020,Elf3Price], Answer),
                 member([elf6,Elf6Textbook,Elf6Year,550],Answer),
                 Elf3Textbook \== Elf6Textbook,
                 Elf3Price \== 550,
                 Elf6Year \== 2020.







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ethics of Present Delivery costs less than the textbook that was published in 2017.
%
%
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate



ethicCostLessThanTextbook2017(Answer) :-

    member([_,textbook4,Textbook4Year,Textbook4Price],Answer),
    member([_,Textbook2017,2017,Prices2017],Answer),
    Textbook4Year \== 2017,
    Textbook2017 \== textbook4,
    prices(PossiblePrices),
    member(Prices2017,PossiblePrices),
    member(Textbook4Price,PossiblePrices),
    Prices2017 > Textbook4Price.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The textbook from 2019 costs $100.00 more than Scheme for Smarties.
%
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate


textbook2019Cost100MoreScheme(Answer) :- member([_,textbook3,Textbook3Year,Textbook3Price], Answer),
                 member([_,Textbook2019,2019,Prices2019],Answer),
                 Textbook3Year \== 2019,
                 Textbook2019 \== textbook3,
                 prices(PossiblePrices),
                 member(Prices2019,PossiblePrices),
                 member(Textbook3Price,PossiblePrices),
                 Prices2019 is 100 + Textbook3Price.




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The textbook that was published in 2018 was written by Missie Toe.
%
%
%
%
textbook2018ByMissie([_,_,[elf3,_,2018,_] |_]).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   The textbook written by Pepper Minstix costs $100.00 less than the textbook that was
%   published in 2014.
%
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate

textbookPepperCost100LessThanTextbook2014(Answer):- member([elf4,TextbookElf4,Elf4Year,Elf4Price], Answer),
                 member([_,Textbook2014,2014,Prices2014],Answer),
                 Elf4Year \== 2014,
                 TextbookElf4 \== Textbook2014,
                 prices(PossiblePrices),
                 member(Prices2014,PossiblePrices),
                 member(Elf4Price,PossiblePrices),
                 Prices2014 is 100+ Elf4Price.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Alabaster Snowball was not the author of Discrete Math for Toy-Making nor of the
%    textbook  that costs $400.00.
%
%   This predicate will find possbile prices and textbooks to pick out
%   from because it doesn't assign any textbooks or prices just stating
%   which one it cannot be.
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate


alabasterNotAuthorDiscreteNorTextbookPrice400(Answer) :-
    member([elf1,TextbookElf1,_,PriceElf1],Answer),
    textbooks(PossibleTextbooks),
    member(TextbookElf1,PossibleTextbooks),
    not(member(TextbookElf1, [textbook5])),
    prices(PossiblePrices),
    member(PriceElf1,PossiblePrices),
    not(member(PriceElf1,[400])).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Of the textbook, Data Structures for Best Cookies, and the title from 2017, one was
%   written by Pepper Minstix and the other costs $400.00.
%
%
%
%
dataStructureand2017WrittenByPepperOrCost400(Answer) :-  member([elf4,textbook2,Elf4Year,Elf4Price],Answer),
                   member([Author2017,Textbook2017,2017,400],Answer),
                   Elf4Year \== 2017,
                   Elf4Price \== 400,
                   Author2017 \== elf4,
                   Textbook2017 \== textbook2.
dataStructureand2017WrittenByPepperOrCost400(Answer) :- member([elf4,TextbookElf4, 2017,Elf4Price],Answer),
                  member([Author400,textbook2,Year400,400],Answer),
                   Year400 \== 2017,
                   Elf4Price \== 400,
                   Author400 \== elf4,
                   TextbookElf4 \== textbook2.




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The textbook that was published in summer 2020 costs $50.00 more than the one
% written by Shinny Upatree.
%
%
%  Parameter: Answer - is a list of lists that has all the elfs
%  instantiated with 3 blanks for each list. The purpose is to fill in
%  the blank the following predicate


textbook2020Cost50LessThanShinnyTextbook(Answer) :-
                 member([elf5,TextbookElf5,Elf5Year,Elf5Price], Answer),
                 member([_,Textbook2020,2020,Prices2020],Answer),
                 Elf5Year \== 2020,
                 TextbookElf5 \== Textbook2020,
                 prices(PossiblePrices),
                member(Prices2020,PossiblePrices),
                 member(Elf5Price,PossiblePrices),
                 Elf5Price is  Prices2020 - 50.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This predicate is to factor out the duplicate atoms or items in a list
%
%
%
%

noDuplicate(Answer) :- member([elf1, Textbook1, Year1,Price1],Answer),
                       member([elf2, Textbook2, Year2,Price2] ,Answer),
                       member([elf3, Textbook3, Year3,Price3] ,Answer),
                       member([elf4, Textbook4, Year4,Price4], Answer),
                       member([elf5, Textbook5, Year5,Price5], Answer),
                       member([elf6, Textbook6, Year6,Price6] , Answer),
                      member([elf7, Textbook7, Year7,Price7] , Answer),

                      Textbook1 \== Textbook2,
                      Textbook1 \== Textbook3,
                      Textbook1 \== Textbook4,
                      Textbook1 \== Textbook5,
                      Textbook1 \== Textbook6,
                      Textbook1 \== Textbook7,

                      Textbook2 \== Textbook3,
                      Textbook2 \== Textbook4,
                      Textbook2 \== Textbook5,
                      Textbook2 \== Textbook6,
                      Textbook2 \== Textbook7,

                      Textbook3 \== Textbook4,
                      Textbook3 \== Textbook5,
                      Textbook3 \== Textbook6,
                      Textbook3 \== Textbook7,

                      Textbook4 \== Textbook5,
                      Textbook4 \== Textbook6,
                      Textbook4 \== Textbook7,

                      Textbook5 \== Textbook6,
                      Textbook5 \== Textbook7,

                      Textbook6 \== Textbook7,


                      Year1 \== Year2,
                      Year1 \== Year3,
                      Year1 \== Year4,
                      Year1 \== Year5,
                      Year1 \== Year6,
                      Year1 \== Year7,

                      Year2 \== Year3,
                      Year2 \== Year4,
                      Year2 \== Year5,
                      Year2 \==Year6,
                      Year2 \== Year7,

                      Year3 \== Year4,
                      Year3 \== Year5,
                      Year3 \== Year6,
                      Year3 \== Year7,

                      Year4 \== Year5,
                      Year4 \== Year6,
                      Year4 \== Year7,

                      Year5 \== Year6,
                      Year5 \== Year7,

                      Year6 \== Year7,


                      Price1 \== Price2,
                      Price1 \== Price3,
                      Price1 \== Price4,
                      Price1 \== Price5,
                      Price1 \== Price6,
                      Price1 \== Price7,

                      Price2 \== Price3,
                      Price2 \== Price4,
                      Price2 \== Price5,
                      Price2 \==Price6,
                      Price2 \== Price7,

                      Price3 \== Price4,
                      Price3 \== Price5,
                      Price3 \== Price6,
                      Price3 \== Price7,

                      Price4 \== Price5,
                      Price4 \== Price6,
                      Price4 \== Price7,

                      Price5 \== Price6,
                      Price5 \== Price7,

                      Price6 \== Price7.











