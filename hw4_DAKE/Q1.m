clear; close all; clc;

%%
% Let B be the dominant allele that codes for brown eyes and b be the
% recessive allele that codes for blue eyes. We are given that the male
% chimpanzee has blue eyes. Hence the genotype of the male chimpanzee is
% bb. The female chimpanzee has brown eyes and hence she can be either BB
% or Bb genotypically. We are given that both of these are equally likely.
% For the sake of consistency, let's keep prefixes of m for male, f for 
% female and c for child at the end of genotypes.
%%
% Hence we have:
%%
% $$P(bbm) = 1 $$; $$P(BBf) = P(Bbf) = 0.5 $$;
%%
% We are also given that the a priori probabilities of the child having any
% of the four possible combinations of genotypes is the same. Hence we
% have:
%%
% $$P(BBc) = P(Bbc) = P(bBc) = P(bbc) = 0.25 $$
%%
% Since Bbc and bBc are the exact same genotypes, we have:
% $$P(BBc) = P(bbc) = 0.25 $$; $$P(Bbc) = 0.5 $$;

%% a)
% We are given that the child has brown eyes and we are to compute the
% likelihood that the female chimp has a blue-eyed gene i.e. that the
% genotype of the female chimp is Bb. Since the male chimp is genotypically
% bb, the phenotype of the child is determined by which allele it receives
% from the mother. If it receives allele B, then the genotype of the child
% becomes Bb and the phenotype is brown eyes. On the other hand, if it
% receives allele b, then the genotype of the child becomes bb and the
% phenotype is blue eyes. Hence the desired probability is:
%%
% $P(Bbm|Bc)$
%%
% Using Bayes' rule, we have:
%%
% $P(Bbm|Bc) = \frac{P(Bc|Bbm)P(Bbm)}{P(Bc)}$
%%
% The probability of the child getting B allele if the mother is
% genotypically Bb is 1/2 and the child will have allele B if the genotype
% of the child is either BB or Bb. Hence P(Bc) = 1/4 + 1/2 = 3/4
%%
% Substituting the values, we get:
%%
% $P(Bbm|Bc) = \frac{1/2\times 1/2}{3/4}$
%%
% $P(Bbm|Bc) = \frac{1}{3}$

%% b)
% Let B1 be the event that the first child has brown eyes and B2 be the 
% event that the second child has brown eyes. Therefore, the probability of
% the mother having b allele i.e. having Bb genotype is given by:
%%
% $P(Bbm|B1c \cap B2c)$
%%
% Using Bayes' rule, we have:
%%
% $P(Bbm|B1c \cap B2c) = \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c)}$
%%
% Similarly, the probability of the mother not carrying b allele i.e.
% having BB genotype is given by:
%%
% $P(BBm|B1c \cap B2c)$
%%
% Using Bayes' rule, we have:
%%
% $P(BBm|B1c \cap B2c) = \frac{P(B1c \cap B2c|BBm)P(BBm)}{P(B1c \cap B2c)}$
%%
% Taking the ratio of the two conditional probabilities:
%%
% $\frac{P(Bbm|B1c \cap B2c)}{P(BBm|B1c \cap B2c)} = \frac{\frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c)}}{\frac{P(B1c \cap B2c|BBm)P(BBm)}{P(B1c \cap B2c)}}$
%%
% Therefore,
%%
% $\frac{P(Bbm|B1c \cap B2c)}{P(BBm|B1c \cap B2c)} = \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}$
%%
% And since these are the only two possible genotypes for the mother, we
% have:
%%
% $$P(Bbm|B1c \cap B2c) + P(BBm|B1c \cap B2c) = 1 $$
%%
% Therefore,
%%
% $$P(BBm|B1c \cap B2c) = 1 - P(Bbm|B1c \cap B2c) $$
%%
% Substituting back into the formula above, we get:
%%
% $\frac{P(Bbm|B1c \cap B2c)}{1 - P(Bbm|B1c \cap B2c)} = \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c) = (1 - P(Bbm|B1c \cap B2c))\times \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c) = \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)} - P(Bbm|B1c \cap B2c)\times \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c) + P(Bbm|B1c \cap B2c)\times \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)} = \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c)\times (1 + \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}) = \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c) = \frac{\frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}}{1 + \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c) = \frac{\frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}}{\frac{P(B1c \cap B2c|BBm)P(BBm) + P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm)}}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c) = \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c|BBm)P(BBm) + P(B1c \cap B2c|Bbm)P(Bbm)}$
%%
% This is basically the Bayes' rule.
%%
% The probability of one child having brown eyes is independent of the
% other given the genotype of the mother. Hence we have:
%%
% $P(B1c \cap B2c|Bbm) = P(B1c|Bbm) \times P(B2c|Bbm) = \frac{1}{2}\times \frac{1}{2} = \frac{1}{4}$
%%
% And we also have:
%%
% $P(B1c \cap B2c|BBm) = P(B1c|BBm) \times P(B2c|BBm) = 1\times 1 = 1$
%%
% Substituting this in the formula, we get:
%%
% $P(Bbm|B1c \cap B2c) = \frac{1/4 \times 1/2}{1\times 1/2 + 1/4\times 1/2}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c) = \frac{1/4}{1 + 1/4} = \frac{1}{5}$

%% c)
% Instead of 2 children, if there were N children with brown eyes, then
% using the same logic as in (b), we have:
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap Bnc) = \frac{P(B1c \cap B2c \cap ... \cap Bnc|Bbm)P(Bbm)}{P(B1c \cap B2c \cap ... \cap Bnc|BBm)P(BBm) + P(B1c \cap B2c \cap ... \cap Bnc|Bbm)P(Bbm)}$
%%
% Because: $$P(Bbm) = P(BBm) = 1/2 $$, we can reduce the equation to:
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap Bnc) = \frac{P(B1c \cap B2c \cap ... \cap Bnc|Bbm)}{P(B1c \cap B2c \cap ... \cap Bnc|BBm) + P(B1c \cap B2c \cap ... \cap Bnc|Bbm)}$
%%
% Also given the genotype of the mother, the genotypes of each child is
% independent of others, hence we have:
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap Bnc) = \frac{(P(B1c|Bbm))^N}{(P(B1c|BBm))^N + (P(B1c|Bbm))^N}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap Bnc) = \frac{(1/2)^N}{(1)^N + (1/2)^N}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap Bnc) = \frac{1/2^N}{1 + 1/2^N}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap Bnc) = \frac{1}{2^N + 1}$