%%
%% Este é o arquivo `tcc_modelo.cls'
%%
%% Esta é a classe de documento `tcc_modelo' usada para composição do LaTeX
%% monografias/Trabalhos de Conclusão de Curso
%% para o curso de Engenharia Computacional da Universidade Federal de Juiz de Fora
%% Copyright (C) 2024 Ruy Freitas Reis
%% 
%% Este programa é software livre; você pode redistribuí-lo e/ou modificar
%% sob os termos da Licença Pública Geral GNU versão 3 como
%% publicado pela Free Software Foundation.
%% 
%% Este programa é distribuído na esperança de que seja útil,
%% mas SEM QUALQUER GARANTIA; mesmo sem a garantia implícita de
%% COMERCIALIZAÇÃO ou ADEQUAÇÃO A UM DETERMINADO FIM. Veja o
%% Licença Pública Geral GNU versão 3 para mais detalhes.
%% 
%% Você deveria ter recebido uma cópia da Licença Pública Geral GNU
%% versão 3 junto com este pacote (veja arquivo COPYING).
%% Caso contrário, consulte <http://www.gnu.org/licenses/>.
%% 
%% Autor(es): Ruy Freitas Reis
%% 
%% \CheckSum{1311}
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%
\def\filename{tcc_modelo.dtx}
\def\fileversion{v2.0}
\def\filedate{2007/03/01}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{tcc_modelo}[\filedate\ \fileversion\ Trabalho de Conclusão de Curso]
\LoadClass[12pt,a4paper,oneside]{book}
\RequirePackage{lastpage}
\RequirePackage{hyperref}
\RequirePackage[english,brazil]{babel}
\RequirePackage{hyphenat}
\RequirePackage{ifthen}
\RequirePackage{graphicx}
\RequirePackage{tabularx}
\RequirePackage{eqparbox}
\RequirePackage[a4paper,bindingoffset=0.0cm,vcentering=true,%
top=30mm,bottom=20mm,left=30mm,right=20mm]{geometry}
\def\mmcTeX{{\rm C\kern-.05em{\sc o\kern-.025em p\kern-.025em
p\kern-.025em e}}\kern-.08em\TeX}
\renewcommand\baselinestretch{1.5}
\let\@adname\@empty
\let\@adsurn\@empty
\let\@coadname\@empty
\let\@coadsurn\@empty

\newif\if@appendix\@appendixtrue

\newcounter{minexaminers}
\newboolean{isdraft}
\newboolean{bancaorientadores}
\newboolean{maledoc}
\setboolean{maledoc}{true}
\setboolean{isdraft}{false}
\newboolean{needbb}
\setboolean{needbb}{false}
\DeclareOption{tcc}{%
  \newcommand{\@degree}{B.Sc.}
  \newcommand{\@degreename}{Bacharelado}
  \newcommand{\local@degname}{Bacharel}
  \newcommand{\foreign@degname}{bachelor}
  \newcommand\local@doctype{Trabalho de Conclus\~{a}o de Curso}
  \newcommand\foreign@doctype{Final Paper}
  \setcounter{minexaminers}{2}
  \setboolean{bancaorientadores}{true}
}
\DeclareOption{draft}{%
  \setboolean{isdraft}{true}
  \ClassWarning{tcc_modelo}{draft mode is ON}%
}
\DeclareOption{dvips}{%
  \PassOptionsToPackage{dvips}{hyperref,graphicx}
}
\DeclareOption{dvipdfm}{%
  \PassOptionsToPackage{dvipdfm}{hyperref,graphicx}
  \setboolean{needbb}{true}
}
\DeclareOption{pdftex}{%
  \PassOptionsToPackage{pdftex}{hyperref,graphicx}
  \PassOptionsToPackage{plainpages=false,pdfpagelabels}{hyperref}
}

\ProcessOptions\relax
\newcommand\university[1]{%
  \renewcommand\local@universityname{#1}
}
\newcommand\course[1]{%
     \ifthenelse{\equal{#1}{ENGCOMP}}
    {\global\def\local@gradname{Engenharia Computacional}
     \global\def\foreign@gradname{Computational Engeneering}}{}
}
\newcommand\institute[1]{%
     \ifthenelse{\equal{#1}{FACENG_ICE}}
    {\global\def\local@instname{Faculdade de Engenharia e Instituto de Ciências Exatas}
    \global\def\local@instabrv{ICE/Engenharia}
     \global\def\foreign@instname{Faculty of Engineering and Institute of Exact Sciences}}{}
}
\renewcommand\title[1]{%
  \global\def\local@title{#1}%
}
\newcommand\foreigntitle[1]{%
  \global\def\foreign@title{#1}%
}
\newcommand\advisor[5]{%
  \global\def\@adtitle{#1}%
  \global\def\@adname{#2}%
  \global\def\@adsurn{#3}%
  \global\def\@addegree{#4}%
  \global\def\@adinst{#5}%
}
\newcommand\coadvisor[5]{%
  \global\def\@coadtitle{#1}%
  \global\def\@coadname{#2}%
  \global\def\@coadsurn{#3}%
  \global\def\@coaddegree{#4}%
  \global\def\@coadinst{#5}%
  %\renewcommand\local@advisorstring{Orientadores}%
  %\renewcommand\foreign@advisorstring{Advisors}%
}

\newcommand\coadvisorII[5]{%
  \global\def\@coadIItitle{#1}%
  \global\def\@coadIIname{#2}%
  \global\def\@coadIIsurn{#3}%
  \global\def\@coadIIdegree{#4}%
  \global\def\@coadIIinst{#5}%
  %\renewcommand\local@advisorstring{Orientadores}%
  %\renewcommand\foreign@advisorstring{Advisors}%
}
\newcommand\examinerI[4]{%
  \global\def\@examItitle{#1}%
  \global\def\@examIname{#2}%
  \global\def\@examIdegree{#3}%
  \global\def\@examIinst{#4}%
}

\newcommand\examinerII[4]{%
  \global\def\@examIItitle{#1}%
  \global\def\@examIIname{#2}%
  \global\def\@examIIdegree{#3}%
  \global\def\@examIIinst{#4}%
}
\newcommand\examinerIII[4]{%
  \global\def\@examIIItitle{#1}%
  \global\def\@examIIIname{#2}%
  \global\def\@examIIIdegree{#3}%
  \global\def\@examIIIinst{#4}%
}
\newcommand\examinerIV[4]{%
  \global\def\@examIVtitle{#1}%
  \global\def\@examIVname{#2}%
  \global\def\@examIVdegree{#3}%
  \global\def\@examIVinst{#4}%
}
\newcount\@examiner\@examiner0
\newcommand\examiner[4]{%
  \global\@namedef{S:\expandafter\the\@examiner}{#1\ #3\ #2\linebreak #4}
  \global\advance\@examiner by 1
}
\renewcommand\author[2]{%
  \global\def\@authname{#1}
  \global\def\@authsurn{#2}
}
\renewcommand\date[3]{%
  \day=#1
  \month=#2
  \year=#3
}
\newcommand\local@monthname{\ifcase\month\or
  janeiro\or fevereiro\or mar{\c c}o\or abril\or maio\or junho\or
  julho\or agosto\or setembro\or outubro\or novembro\or dezembro\fi}
\newcommand\foreign@monthname{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi}
\newcommand\place[3]{%
  \renewcommand\local@cityname{#1}
  \renewcommand\local@statename{#2}
  \renewcommand\local@countryname{#3}
}
\newcounter{keywords}
\newcommand\keyword[1]{%
  \global\@namedef{K:\expandafter\the\c@keywords}{#1}
  \global\addtocounter{keywords}{1}
}
\newcounter{keywordsEN}
\newcommand\keywordEN[1]{%
  \global\@namedef{P:\expandafter\the\c@keywordsEN}{#1}
  \global\addtocounter{keywordsEN}{1}
}
\renewcommand\frontmatter{%
  \cleardoublepage
  \@mainmatterfalse
 \@appendixfalse
%   \pagenumbering{roman}
  \thispagestyle{empty}
%   \setcounter{page}{2}
  \makefrontpage
  \clearpage
%   \pagestyle{plain}
%   \ifthenelse{\boolean{maledoc}}{}{}%
}





\renewcommand\mainmatter{%

  \immediate\write\@auxout{%
    \string\newlabel{front:pageno}{{}{\Roman{page}}{}{page.\roman{page}}{}}}
  \cleardoublepage
  \@mainmattertrue
\@appendixfalse
%   \pagestyle{plain}%numero da pagina no centro do rodape
%   \pagestyle{headings}%numero da pagina no cabecalho+titulo do capitulo
  \pagestyle{myheadings}%numero da pagina no cabecalho



%   \pagenumbering{arabic}


}



\renewcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi}

%   \baselineskip14pt\sloppy\nohyphens{\local@doctype\
%   \ifthenelse{\boolean{maledoc}}{apresentado}{apresentada}
%   ao Programa de P{\' o}s-gradua{\c c}{\~ a}o em \local@gradname,
%   da \local@universityname\ como requisito parcial
%   {\` a} obten{\c c}{\~ a}o do grau de
%   \local@degname\ em \local@gradname.}
%\newenvironment{capa}{\begin{titlingpage}}{\end{titlingpage}\cleardoublepage}

\newcommand{\inserecapa}{%
\pagestyle{empty}
  \begin{titlepage}%
  \begin{center}
    {\local@universityname \par \local@instname  \par Bacharelado em \local@gradname} 
    \vfill
    { \bfseries \nohyphens{\@authname\ \@authsurn}\par}
    \vfill
    {\bfseries\local@title }
    \vfill
    \frontpage@bottomtext
  \end{center}
  \end{titlepage}
}

\renewcommand\maketitle{%
\pagestyle{empty}
% \pagenumbering{alph}
\config@hypersetup
  \begin{titlepage}
%   \begin{center}
%     \baselineskip 2cm 
% \large{UNIVERSIDADE FEDERAL DE JUIZ DE FORA \\
%     {PROGRAMA DE P\'OS-GRADUA\c C\~AO EM MODELAGEM COMPUTACIONAL}} \\
%     \end{center}
 % \begin{flushleft}
 % \setlength\baselineskip{0pt}
 % \setlength\parskip{1mm}
 % \makebox[20mm][c]{%
 %   \ifthenelse{\boolean{needbb}}{%
 %     \includegraphics[bb = -1 -1 48 64]{minerva}}{%
 %     \includegraphics{minerva}}}\par
%  \makebox[20mm][c]{{\fontfamily{phv}\textbf{\small MMC/UFJF}}}
%  \end{flushleft}
\begin{center}
\nohyphens{\@authname\ \@authsurn}\par
\vspace*{3.5cm}
%   \nohyphens{\MakeUppercase\local@title}\par
\nohyphens{\textbf{\local@title}}\par
\end{center}
  \vspace*{3cm}
  \begin{flushright}
  \begin{minipage}{80mm}
\frontcover@maintext
  \end{minipage}\par
%   \vspace*{7.5mm}
%   \begin{tabularx}{80mm}[b]{@{}l@{ }X@{}}
%     \local@advisorstring: & \nohyphens{\@adname\ \@adsurn}\\
%     \ifthenelse{\equal{\@coadname}{\@empty}}{}{%
%     Coorientador: & \nohyphens{\@coadname\ \@coadsurn}\\}
%     \ifthenelse{\equal{\@coadIIname}{\@empty}}{}{%
%     Coorientador: & \nohyphens{\@coadIIname\ \@coadIIsurn}\\}
%   \end{tabularx}\par
  \end{flushright}
  \vspace*{2.7cm}
  \begin{tabularx}{100mm}[b]{@{}l@{ }X@{}}
    \local@advisorstring: & \nohyphens{\@adtitle\ \@addegree\ \@adname\ \@adsurn}\\
    \ifthenelse{\equal{\@coadname}{\@empty}}{}{%
    Coorientador: & \nohyphens{\@coadtitle\ \@coaddegree\ \@coadname\ \@coadsurn}\\}
    \ifthenelse{\equal{\@coadIIname}{\@empty}}{}{%
    Coorientador: & \nohyphens{\@coadIItitle\ \@coadIIdegree\ \@coadIIname\ \@coadIIsurn}\\}
  \end{tabularx}\par
  \vspace*{\fill}
  \begin{center}
%   \local@cityname\par
% %   \local@monthname\ de \number\year
%   \number\year
  \vspace*{\fill}
  \frontpage@bottomtext
  \end{center}
  \end{titlepage}
  \global\let\maketitle\relax%
  \global\let\and\relax


\setcounter{page}{2}
\makecatalog
%\includepdf[pages=1]{ficha_catalografica.pdf}

}




\newcommand\makefrontpage{%
%\includepdf[pages=1]{frontpage.pdf}

  \begin{center}
  \nohyphens{\@authname\ \@authsurn}\par
  \vspace*{1.5cm}
%   \nohyphens{\MakeUppercase\local@title}\par
  \nohyphens{\textbf{\local@title}}\par

  \end{center}
  \vspace*{1.cm}
  \begin{flushright}
  \begin{minipage}{80mm}
  \frontcover@maintext
  \end{minipage}\par
  \end{flushright}
  \vspace*{1.cm}

%   \begin{center}
%     \sloppy\nohyphens{\MakeUppercase\local@title}\par
%     \vspace*{7mm}
%     {\@authname\ \@authsurn}\par
%   \end{center}\par
%   \vspace*{4mm}
%   \frontpage@maintext
%   \vspace*{8mm}


  \noindent\local@approvedname \ \number\day \ de \local@monthname \ de \number\year.\par






     \begin{center}
BANCA EXAMINADORA
%   \begin{flushright}
%\begin{multicols}{2}
%\footnotesize
\ifthenelse{\boolean{bancaorientadores}}{
	\baselineskip14pt
        \vskip20\p@ 
        \eqparbox{signture@list}{\hrulefill}\\
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@adtitle\ \@addegree\ \@adname\ \@adsurn\ -\ Orientador\ \ \ } \\ 
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@adinst\ \ \ } \\ 
        \ifthenelse{\equal{\@coadname}{\@empty}}{}{%
        \vskip20\p@ %\baselineskip12pt
        \eqparbox{signture@list}{\hrulefill}\\
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@coadtitle\ \@coaddegree\ \@coadname\ \@coadsurn \ -\ Coorientador\ }\\ 
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@coadinst \ \ \ }
\\}
	\ifthenelse{\equal{\@coadIIname}{\@empty}}{}{%
        \vskip20\p@ %\baselineskip12pt
        \eqparbox{signture@list}{\hrulefill}\\
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@coadIItitle\ \@coadIIdegree\ \@coadIIname\ \@coadIIsurn \ -\ Coorientador\ } \\ 
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@coadIIinst \ \ \ }
\\}
}{}
        \ifthenelse{\equal{\@examIname}{\@empty}}{}{%
        \vskip30\p@ %\baselineskip12pt
        \eqparbox{signture@list}{\hrulefill}\\
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@examItitle\ \@examIdegree\ \@examIname\ \ \ }\\ 
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@examIinst \ \ \ }
\\}
        \ifthenelse{\equal{\@examIIname}{\@empty}}{}{%
        \vskip30\p@ %\baselineskip12pt
        \eqparbox{signture@list}{\hrulefill}\\
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@examIItitle\ \@examIIdegree\ \@examIIname\ \ \ }\\ 
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@examIIinst \ \ \ }
\\}
        \ifthenelse{\equal{\@examIIIname}{\@empty}}{}{%
        \vskip30\p@ %\baselineskip12pt
        \eqparbox{signture@list}{\hrulefill}\\
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@examIIItitle\ \@examIIIdegree\ \@examIIIname\ \ \ }\\ 
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@examIIIinst \ \ \ }
\\}
        \ifthenelse{\equal{\@examIVname}{\@empty}}{}{%
        \vskip30\p@ %\baselineskip12pt
        \eqparbox{signture@list}{\hrulefill}\\
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@examIVtitle\ \@examIVdegree\ \@examIVname\ \ \ }\\ 
        \eqparbox{signture@list}{\protect\centering%
          \ \ \ \@examIVinst \ \ \ }
\\}
%\end{multicols}
%         \count1=0
%         \@whilenum \count1<\@examiner \do {
%             \vskip40\p@ \baselineskip12pt
%         \eqparbox{signture@list}{\hrulefill}\\
%         \eqparbox{signture@list}{\protect\centering%
%             \baselineskip18pt\ \ \ \csname S:\the\count1 \endcsname\ \ \ }\\
%             \advance\count1 by 1}



%   \end{flushright}

     \end{center}
%   \vspace*{\fill}
%   \frontpage@bottomtext
}  %end fontpage
\newcommand\config@hypersetup{%
\begingroup
  % changes to \toks@ and \count@ are kept local;
  % it's not necessary for them, but it is usually the case
  % for \count1, because the first ten counters are written
  % to the DVI file, thus you got lucky because of PDF output
  \toks@={}% in this special case not necessary
  \count@=0 %
  \@whilenum\count@<\value{keywords}\do{%
    % * a keyword separator is not necessary,
    %    if there is just one keyword
    % * \csname K:\the\count@\endcsname must be expanded
    %    at least once, to get rid of the loop depended \count@
    \ifcase\count@ % same as \ifnum0=\count@
      \toks@=\expandafter{\csname K:\the\count@\endcsname}%
    \else
      \toks@=\expandafter\expandafter\expandafter{%
        \expandafter\the\expandafter\toks@
        \expandafter;\expandafter\space
        \csname K:\the\count@\endcsname
      }%
    \fi
    \advance\count@ by 1 %
  }%
  \edef\x{\endgroup
    \noexpand\hypersetup{%
      pdfkeywords={\the\toks@}%
    }%
  }%
\x
\hypersetup{%
  pdfauthor={\@authname\ \@authsurn},
  pdftitle={\local@title},
 pdfsubject={\local@doctype\ de \@degreename\ em \local@gradname\ da EngComp/UFJF},
  pdfcreator={LaTeX with CoppeTeX toolkit},
  breaklinks={true},
  raiselinks={true},
  pageanchor={true},
}}
\newcommand\makecatalog{%
  \pagestyle{empty}
  \vspace*{\fill}
  \begin{center}
    \setlength{\fboxsep}{5mm}
    \framebox[120mm][c]{\makebox[5mm][c]{}%
      \begin{minipage}[c]{105mm}
      \setlength{\parindent}{5mm}
      \noindent\sloppy\nohyphens\@authsurn,
      \nohyphens\@authname\par
      \nohyphens{\local@title/\@authname\ \@authsurn. - - \number\year.}\par
      \pageref{front:pageno},
      \pageref{LastPage}
      p.\@ifundefined{r@cat:lofflag}{}{\pageref{cat:lofflag}} $29,7$cm.\par
      % There is an issue here. When the last entry must be split between lines,
      % the spacing between it and the next paragraph becomes smaller.
      % Should we manually introduce a fixed space? But how could we know that
      % a name was split?
      \begin{tabularx}{100mm}[b]{@{}l@{ }X@{}}
            \local@advisorstring: & \nohyphens{\@adname\ \@adsurn}\\
	    \ifthenelse{\equal{\@coadname}{\@empty}}{}{%
	    Coorientador: & \nohyphens{\@coadname\ \@coadsurn}\\}
	    \ifthenelse{\equal{\@coadIIname}{\@empty}}{}{%
	    Coorientador: & \nohyphens{\@coadIIname\ \@coadIIsurn}\\}
      \end{tabularx}\par
      \nohyphens{\local@doctype\ ({\MakeLowercase\@degreename}) --
      \local@universityname, \local@instabrv. \@degreename \ em \  \local@gradname , \number\year.}\par
      \bibname: p. \pageref{bib:begin} -- \pageref{bib:end}.\par
      \count1=0
      \count2=1
      \nohyphens{\@whilenum \count1<\value{keywords} \do {%
        \number\count2. \csname K:\the\count1 \endcsname.
        \advance\count1 by 1
        \advance\count2 by 1}
      I. \@adsurn, \@adname%
      \ifthenelse{\equal{\@coadname}{\@empty}}{}{\emph{~et~al.}}.
      II. \local@universityname, Bacharelado em \local@gradname.}
    \end{minipage}}
  \end{center}
  \vspace*{\fill}}
% \newcommand\dedication[1]{
%   \gdef\@dedic{#1}
%     \cleardoublepage
%     \vspace*{\fill}
%     \begin{flushright}
%       \begin{minipage}{60mm}
%         \raggedleft \it \normalsize \@dedic
%       \end{minipage}
%     \end{flushright}}
\newcommand\dedication[1]{
  \gdef\@dedic{#1}
    \cleardoublepage
    \vspace*{\fill}
    \begin{flushright}
      \begin{minipage}{60mm}
        \it \normalsize \@dedic
      \end{minipage}
    \end{flushright}}

% \newcommand\epigrafe[1]{
%   \gdef\@dedic{#1}
%     \cleardoublepage
%     \vspace*{\fill}
%     \begin{flushright}
%       \begin{minipage}{60mm}
%         \raggedleft \it \normalsize \@dedic
%       \end{minipage}
%     \end{flushright}}
\newcommand\epigrafe[1]{
  \gdef\@dedic{#1}
    \cleardoublepage
    \vspace*{\fill}
    \begin{flushright}
      \begin{minipage}{60mm}
         \it \normalsize \@dedic
      \end{minipage}
    \end{flushright}}



\renewcommand\tableofcontents{%
\clearpage
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
%     \section*{\centering\MakeUppercase\contentsname
%         \@mkboth{%
%            \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    {\begin{center}\textbf{\MakeUppercase{\contentsname}}\end{center}
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}\vspace*{1.cm}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }


% \setcounter{secnumdepth}{2}
\setcounter{secnumdepth}{3}  %para numerar ate subsubsection



\renewcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{4.1em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries 
      \advance\leftskip\@tempdima 
      \hskip -\leftskip
      #1\nobreak\hfil \dotfill\nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}


\renewcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\m@ne
%     \addpenalty{-\@highpenalty}%
%     \vskip 1.0em \@plus\p@
    \setlength\@tempdima{4.1em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode
      \advance\leftskip\@tempdima 
      \hskip -\leftskip
      #1 \nobreak\hfil \dotfill\nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}


\renewcommand*\l@subsection[2]{%
  \ifnum \c@tocdepth >\m@ne
%     \addpenalty{-\@highpenalty}%
%     \vskip 1.0em \@plus\p@
    \setlength\@tempdima{4.1em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima 
      \hskip -\leftskip
      #1\nobreak\hfil \dotfill\nobreak\hb@xt@\@pnumwidth{\bfseries\scshape\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}


\renewcommand*\l@subsubsection[2]{%
  \ifnum \c@tocdepth >\m@ne
%     \addpenalty{-\@highpenalty}%
%     \vskip 1.0em \@plus\p@
    \setlength\@tempdima{4.1em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \mdseries\itshape 
      \advance\leftskip\@tempdima 
      \hskip -\leftskip
      #1\nobreak\hfil \dotfill\nobreak\hb@xt@\@pnumwidth{\bfseries\scshape\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}

% \renewcommand*\l@section{\bfseries\@dottedtocline{1}{0em}{2.3em}}
% \renewcommand*\l@subsection{\@dottedtocline{2}{0em}{3.2em}}
% \renewcommand*\l@subsubsection{\@dottedtocline{3}{0em}{4.1em}}
\renewcommand*\l@paragraph{\@dottedtocline{4}{0em}{5em}}
\renewcommand*\l@subparagraph{\@dottedtocline{5}{0em}{6em}}



\renewcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
%                     \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
		    

                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%

			  \if@appendix
% 			      \addcontentsline{toc}{chapter}% NAO COLOCA NO SUMARIO OS CAPITULOS DO APENDICE
%                                  {\protect\numberline{\thechapter}ddd#1}%
			  \else
				\addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
			  \fi


%                          \addcontentsline{toc}{chapter}%
%                                    {\protect\numberline{\thechapter}#1}%
                       \else
			  
				\addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\def\@makechapterhead#1{%
%   \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
%         \huge\bfseries \@chapapp\space \thechapter
%         \Huge\bfseries \thechapter

        \par\nobreak
        \vskip 20\p@
      \fi
    \fi
    \interlinepenalty\@M

    \if@appendix
	\huge \bfseries \MakeUppercase{\appendixname} \thechapter\ - #1\par\nobreak
    \else
% 	\huge \bfseries \space \space #1\par\nobreak
        \huge \bfseries \thechapter \space \space #1\par\nobreak
    \fi

    \vskip 40\p@
  }}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries   #1\par\nobreak
    \vskip 40\p@
  }}
  


\renewcommand\chapter{\cleardoublepage\@startsection {chapter}{0}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\normalsize\bfseries}}
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\normalsize\MakeUppercase}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\itshape}}




\newenvironment{agradecimentos}{%
  \clearpage
%   \thispagestyle{plain}
  \pagestyle{empty}
\begin{center}
\textbf{AGRADECIMENTOS}
\end{center}
\vspace*{1.cm}
\par
% \begin{center}
%  \section*{\centering AGRADECIMENTOS}
% \end{center}
}







\newenvironment{abstract}{%
  \clearpage
%   \thispagestyle{plain}
  \pagestyle{empty}
  \abstract@toptext
\vspace*{1.cm}
%   \vspace*{8.6mm}
%   \begin{center}
%     \sloppy\nohyphens{\MakeUppercase\local@title}\par
%     \vspace*{13.2mm}
%     \@authname\ \@authsurn \par
%     \vspace*{7mm}
%     \local@monthname/\number\year
%   \end{center}\par
%   \vspace*{\fill}
%   \noindent%
%   \begin{tabularx}{\textwidth}[b]{@{}l@{ }X@{}}
%     \local@advisorstring: & \nohyphens{\@adname\ \@adsurn}\\
%     \ifthenelse{\equal{\@coadname}{\@empty}}{}{%
%     Coorientador: & \nohyphens{\@coadname\ \@coadsurn}\\}
%     \ifthenelse{\equal{\@coadIIname}{\@empty}}{}{%
%     Coorientador: & \nohyphens{\@coadIIname\ \@coadIIsurn}\\}
%   \end{tabularx}\par
%   \vspace*{2mm}
%   \noindent\local@deptstring: \local@gradname\par
%   \vspace*{7mm}
}{\vspace*{1.5cm} \\
\textbf{Palavras-chave:} \count1=0 \nohyphens{\@whilenum \count1<\value{keywords} \do {%
        \csname K:\the\count1 \endcsname.
        \advance\count1 by 1}}
\vspace*{\fill}}




\newenvironment{foreignabstract}{%
  \clearpage
%   \thispagestyle{plain}
  \pagestyle{empty}
  \begin{otherlanguage}{english}
  \foreignabstract@toptext
\vspace*{1.cm}
%   \vspace*{8.6mm}
%   \begin{center}
%     \sloppy\nohyphens{\MakeUppercase\foreign@title}\par
%     \vspace*{13.2mm}
%     \@authname\ \@authsurn \par
%     \vspace*{7mm}
%     \foreign@monthname/\number\year
%   \end{center}\par
%   \vspace*{\fill}
%   \noindent%
%   \begin{tabularx}{\textwidth}[b]{@{}l@{ }X@{}}
%     \foreign@advisorstring: & \nohyphens{\@adname\ \@adsurn}\\
%     \ifthenelse{\equal{\@coadname}{\@empty}}{}{%
%     Co-advisor: & \nohyphens{\@coadname\ \@coadsurn}\\}
%     \ifthenelse{\equal{\@coadIIname}{\@empty}}{}{%
%     Co-advisor: & \nohyphens{\@coadIIname\ \@coadIIsurn}\\}
%   \end{tabularx}\par
%   \vspace*{2mm}
%   \noindent\foreign@deptstring: \foreign@gradname\par
%   \vspace*{7mm}
}{%
\vspace*{1.5cm} \\
\textbf{Keywords:} \count1=0 \nohyphens{\@whilenum \count1<\value{keywordsEN} \do {%
        \csname P:\the\count1 \endcsname.
        \advance\count1 by 1}}
  \end{otherlanguage}
  \vspace*{\fill}
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\foreign@title\@empty
  \global\let\foreign@title\relax
  \global\let\local@title\@empty
  \global\let\local@title\relax
  \global\let\author\relax
  \global\let\author\relax
  \global\let\date\relax}



\renewcommand\listoffigures{%
\clearpage
  \pagestyle{empty}
  \immediate\write\@auxout{%
  \string\newlabel{cat:lofflag}{{}{:~il.;}{}{page.\roman{page}}{}}}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
%     \chapter*{\listfigurename}%
% \section*{\centering\MakeUppercase\listfigurename}
    \begin{center}\textbf{\MakeUppercase{lista de ilustra\c{c}\~oes}}\end{center}
%       \addcontentsline{toc}{chapter}{\listfigurename}%
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}
\vspace*{1.cm}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }


\renewcommand\listoftables{%
\clearpage
\pagestyle{empty}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
%     \chapter*{\listtablename}%
% \section*{\centering\MakeUppercase\listtablename}
    \begin{center}\textbf{\MakeUppercase{\listtablename}}\end{center}
%       \addcontentsline{toc}{chapter}{\listtablename}%
      \@mkboth{%
          \MakeUppercase\listtablename}%
         {\MakeUppercase\listtablename}%
\vspace*{1.cm}
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }


\newcommand\printlosymbols{%
\renewcommand\glossaryname{\listsymbolname}%
\@input@{\jobname.los}}
\def\makelosymbols{%
  \newwrite\@losfile
  \immediate\openout\@losfile=\jobname.syx
  \def\symbl{\@bsphack\begingroup
             \@sanitize
             \@wrlos}\typeout
  {Writing index of symbols file \jobname.syx}%
  \let\makelosymbols\@empty
}
\@onlypreamble\makelosymbols
\AtBeginDocument{%
\@ifpackageloaded{hyperref}{%
  \def\@wrlos#1#2{%
    \protected@write\@losfile{}%
      {\string\indexentry{[#1] #2|hyperpage}{\thepage}}%
    \endgroup
    \@esphack}}{%
  \def\@wrlos#1#2{%
    \protected@write\@losfile{}%
      {\string\indexentry{[#1] #2}{\thepage}}%
    \endgroup
    \@esphack}}}


\newcommand\printloabbreviations{%
\renewcommand\glossaryname{\listabbreviationname}%
\@input@{\jobname.lab}}
\def\makeloabbreviations{%
  \newwrite\@labfile
  \immediate\openout\@labfile=\jobname.abx
  \def\abbrev{\@bsphack\begingroup
              \@sanitize
              \@wrlab}\typeout
  {Writing index of abbreviations file \jobname.abx}%
  \let\makeloabbreviations\@empty
}
\@onlypreamble\makeloabbreviations
\AtBeginDocument{%
\@ifpackageloaded{hyperref}{%
  \def\@wrlab#1#2{%
    \protected@write\@labfile{}%
      {\string\indexentry{[#1] #2|hyperpage}{\thepage}}%
    \endgroup
    \@esphack}}{%
  \def\@wrlab#1#2{%
    \protected@write\@labfile{}%
      {\string\indexentry{[#1] #2}{\arabic{page}}}%
    \endgroup
    \@esphack}}}
\newdimen\bibindent%
\setlength\bibindent{1.5em}%
\renewenvironment{thebibliography}[1]%
     {
% 	\chapter*{Refer\^encias}%
% \section*{\centering REFER\^ENCIAS}
   \begin{center}\textbf{\MakeUppercase{REFER\^ENCIAS}}\end{center}
%     	\begin{center}
% 		\section*{Refer\^encias}
% 	\end{center}%
%       \addcontentsline{toc}{chapter}{\bibname}%
      \addcontentsline{toc}{chapter}{REFER\^ENCIAS}%
      \immediate\write\@auxout{%
      \string\newlabel{bib:begin}{{}{\arabic{page}}{}{page.\arabic{page}}{}}}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\setlength{\labelwidth}{0ex}%
            \setlength{\leftmargin}{9.0ex}%
            \setlength{\itemindent}{-9.0ex}%
            \advance\leftmargin\labelsep%
            \@openbib@code%
            \usecounter{enumiv}%
            \let\p@enumiv\@empty%
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy%
      \clubpenalty4000%
      \@clubpenalty \clubpenalty%
      \widowpenalty4000%
      \sfcode`\.\@m}%
     {\def\@noitemerr%
       {\@latex@warning{Empty `thebibliography' environment}}%
      \immediate\write\@auxout{%
      \string\newlabel{bib:end}{{}{\arabic{page}}{}{page.\arabic{page}}{}}}%
      \endlist}


\renewcommand\appendix{
% \clearpage
\@appendixtrue
\addcontentsline{toc}{chapter}{AP\^ENDICES}%
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}
}



\newenvironment{theglossary}{%
  \if@twocolumn%
    \@restonecoltrue\onecolumn%
  \else%
    \@restonecolfalse%
  \fi%
  \@mkboth{\MakeUppercase\glossaryname}%
  {\MakeUppercase\glossaryname}%
%   \chapter*{\glossaryname}%
% \section*{\centering\MakeUppercase\glossaryname}
   \begin{center}\textbf{\MakeUppercase{\glossaryname}}\end{center}
%   \addcontentsline{toc}{chapter}{\glossaryname}
  \list{}
  {\setlength{\listparindent}{0in}%
   \setlength{\labelwidth}{1.0in}%
   \setlength{\leftmargin}{1.5in}%
   \setlength{\labelsep}{0.5in}%
   \setlength{\itemindent}{0in}}%
   \sloppy}%
  {\if@restonecol\twocolumn\fi%
\endlist}
\renewenvironment{theindex}{%
  \if@twocolumn
    \@restonecolfalse
  \else
    \@restonecoltrue
  \fi
  \twocolumn[\@makeschapterhead{\indexname}]%
  \@mkboth{\MakeUppercase\indexname}%
  {\MakeUppercase\indexname}%
  \thispagestyle{plain}\parindent\z@
%   \addcontentsline{toc}{chapter}{\indexname}
  \parskip\z@ \@plus .3\p@\relax
  \columnseprule \z@
  \columnsep 35\p@
  \let\item\@idxitem}
  {\if@restonecol\onecolumn\else\clearpage\fi}

\renewcommand\appendixname{Ap\^endices}
\newcommand\glossaryname{Glossary}
\newcommand\listabbreviationname{Lista de Abreviaturas e Siglas}
\renewcommand\listfigurename{Lista de Ilustra\c{c}\~oes}
\newcommand\listsymbolname{Lista de S{\' i}mbolos}
\newcommand\local@advisorstring{Orientador}
\newcommand\foreign@advisorstring{Advisor}
\newcommand\local@coadvisorstring{Coorientador}
\ifthenelse{\boolean{maledoc}}{%
%   \newcommand\local@approvedname{Aprovado por}%
  \newcommand\local@approvedname{Aprovado em}%
}{%
%   \newcommand\local@approvedname{Aprovada por}%
  \newcommand\local@approvedname{Aprovada em}%
}
\newcommand\foreign@approvedname{Approved by}
\newcommand\local@universityname{Universidade Federal de Juiz de Fora}
\newcommand\local@universityabrv{UFJF}
\newcommand\local@deptstring{Departamento}
\newcommand\foreign@deptstring{Department}
\newcommand\local@cityname{Juiz de Fora}
\newcommand\local@statename{MG}
\newcommand\local@countryname{Brasil}
\newcommand\frontcover@maintext{
%   \sloppy\nohyphens{\local@doctype\ de \@degreename\
   \baselineskip14pt\sloppy\nohyphens{\local@doctype\
  \ifthenelse{\boolean{maledoc}}{apresentado}{apresentada}
  ao corpo docente do curso de graduação em \local@gradname \ da \local@instname 
  \ da \local@universityname\ como requisito parcial
  {\` a} obten{\c c}{\~ a}o do grau de
  \local@degname\ em \local@gradname.}
}



\newcommand\frontpage@maintext{
  \noindent {\MakeUppercase\local@doctype}
  \ifthenelse{\boolean{maledoc}}{SUBMETIDO}{SUBMETIDA}
  \sloppy\nohyphens{AO CORPO DOCENTE DO
  P{\' O}S-GRADUA{\c C}{\~ A}O EM MODELAGEM COMPUTACIONAL DA
  UNIVERSIDADE FEDERAL DE JUIZ DE FORA COMO PARTE DOS REQUISITOS
  NECESS{\' A}RIOS PARA A OBTEN{\c C}{\~ A}O DO GRAU DE
  {\MakeUppercase\local@degname} EM CI{\^E}NCIAS EM}
  {\MakeUppercase\local@gradname. \par}%
}
\newcommand\frontpage@bottomtext{%
  \begin{center}
%   {\MakeUppercase{\local@cityname, \local@statename\ -- \local@countryname}}\par
%   {\MakeUppercase\local@monthname\ DE \number\year}
  \local@cityname\par
  \number\year
  \end{center}%
}
\newcommand\abstract@toptext{%
% \chapter*{Resumo}
% \section*{\centering RESUMO}
\begin{center}\textbf{RESUMO} \end{center}
%   \noindent Resumo \ifthenelse{\boolean{maledoc}}{do}{da}
%   \local@doctype\ \ifthenelse{\boolean{maledoc}}{apresentado}{apresentada}
%   \sloppy\nohyphens{{\` a} UFJF como parte dos requisitos
%   necess{\' a}rios para a obten{\c c}{\~ a}o do grau de
%   \local@degname\ em Ci{\^ e}ncias (\@degree)}
}
\newcommand\foreignabstract@toptext{%
% \chapter*{Abstract}
% \section*{\centering ABSTRACT}
\begin{center}\textbf{ABSTRACT} \end{center}
%   \noindent \sloppy\nohyphens{Abstract of \foreign@doctype\ presented to
%   UFJF as a partial fulfillment of the requirements for the
%   degree of \foreign@degname\ of Science (\@degree)}
}
%% 
%%
%% End of file `tcc_modelo.cls'.
