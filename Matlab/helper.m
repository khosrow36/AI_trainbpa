%odczyt zapisanego pliku
load("output14.mat");
%tworzenie wykresu z PK, analogizcznie robi sie SSE
%surf(PK_v)
%xlabel('S2') 
%ylabel('S1')
%zlabel('PK')

%zapisywanie do excela
T = table(parameters_e);
writetable(T,'e3.xlsx')
