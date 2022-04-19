\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cukproposal}

\newif\ifgerman\germanfalse

\DeclareOption{german}{\germantrue}

\ProcessOptions

\LoadClass[a4paper, 11pt, oneside, numbers=noenddot, bibliography=totoc]{scrartcl}

\ifgerman
	\RequirePackage[ngerman]{babel}
\else
	\RequirePackage[english]{babel}
\fi
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern}

% Some older versions of MiKTeX are still missing the scrlayer-scrpage package ... -.-'
\IfFileExists{scrlayer-scrpage.sty}
{
	\RequirePackage[automark,headsepline]{scrlayer-scrpage}
}{
	\RequirePackage{fancyhdr}
}

% Header and footer Layout depending on the loaded package
\IfFileExists{scrlayer-scrpage.sty}
{
	\pagestyle{scrheadings}

	\clearpairofpagestyles
	\cfoot[\pagemark]{\pagemark}
	\chead{\headmark}
}{
	\pagestyle{fancy}

	\lhead[]{}
	\rhead[]{}
	\chead[]{\rightmark}
	\cfoot{\thepage}

	\fancypagestyle{plain}{
	\fancyhf{}
	\cfoot{\thepage}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
	}
}

\renewcommand{\sectionmark}[1]{\markright{\textsc{#1}}}
\renewcommand{\subsectionmark}[1]{\markright{\textsl{\thesubsection~#1}}}

% Essential math packages
\RequirePackage{amsmath,amsfonts,amssymb,amsthm,mathtools}

% Theorem style
\newtheoremstyle{thmstyle}
  {3pt}{3pt}{\itshape}{}{\bfseries}{}{.5em}
  {\thmname{#1}\thmnumber{ #2}\thmnote{ \textmd{(#3)}}}
\theoremstyle{thmstyle}

% Theorem environments
\newtheorem{definition}{Definition}[section]
\newtheorem{lemma}[definition]{Lemma}
\newtheorem{problem}[definition]{Problem}

\ifgerman
	\newtheorem{proposition}[definition]{Satz}
	\newtheorem{theorem}[definition]{Satz}
	\newtheorem{corollary}[definition]{Korollar}
	\newtheorem{example}[definition]{Beispiel}
	\newtheorem{claim}[definition]{Behauptung}
	\newtheorem{remark}[definition]{Bemerkung}
	\newtheorem{observation}[definition]{Beobachtung}
\else
	\newtheorem{proposition}[definition]{Proposition}
	\newtheorem{theorem}[definition]{Theorem}
	\newtheorem{corollary}[definition]{Corollary}
	\newtheorem{example}[definition]{Example}
	\newtheorem{claim}[definition]{Claim}
	\newtheorem{remark}[definition]{Remark}
	\newtheorem{observation}[definition]{Observation}
\fi
