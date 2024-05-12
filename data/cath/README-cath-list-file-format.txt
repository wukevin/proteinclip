CATH List File (CLF) Format 2.0
-------------------------------
This file format has an entry for each structural entry in CATH.

Column 1:  CATH domain name (seven characters)
Column 2:  Class number
Column 3:  Architecture number
Column 4:  Topology number
Column 5:  Homologous superfamily number
Column 6:  S35 sequence cluster number
Column 7:  S60 sequence cluster number
Column 8:  S95 sequence cluster number
Column 9:  S100 sequence cluster number
Column 10: S100 sequence count number
Column 11: Domain length
Column 12: Structure resolution (Angstroms)
           (999.000 for NMR structures and 1000.000 for obsolete PDB entries)

Comment lines start with a '#' character.

Example:
--------
1oaiA00     1    10     8    10     1     1     1     1     1    59 1.000
1go5A00     1    10     8    10     1     1     1     1     2    69 999.000
1oksA00     1    10     8    10     2     1     1     1     1    51 1.800
1t6oA00     1    10     8    10     2     1     2     1     1    49 2.000
1cuk003     1    10     8    10     3     1     1     1     1    48 1.900
1hjp003     1    10     8    10     3     1     1     2     1    44 2.500
1c7yA03     1    10     8    10     3     1     1     2     2    48 3.100
1p3qQ00     1    10     8    10     4     1     1     1     1    43 1.700
1mn3A00     1    10     8    10     4     1     2     1     1    52 2.300
1nv8B01     1    10     8    10     5     1     1     1     1    71 2.200


CATH Domain Names
-----------------
The domain names have seven characters (e.g. 1oaiA00).

CHARACTERS 1-4: PDB Code
The first 4 characters determine the PDB code e.g. 1oai

CHARACTER 5: Chain Character
This determines which PDB chain is represented.
Chain characters of zero ('0') indicate that the PDB file has no chain field.

CHARACTER 6-7: Domain Number
The domain number is a 2-figure, zero-padded number (e.g. '01', '02' ... '10', '11', '12'). Where the domain number is a double ZERO ('00') this indicates that the domain is a whole PDB chain with no domain chopping. 


Hierachy Node Representatives
-----------------------------
Representative structural domains are selected from the CathDomainList based on 
the numbering scheme. For example the S35 sequence family representatives 
for superfamily 1.10.8.10 in the above example are 1oaiA00, 1oksA00, 1cuk003,
1p3qQ00 and 1nv8B01 as these are the first instances in the file with the same
superfamily number i.e. 1.10.8.10 but all have different S35 numbers (1 to 5).



New to CLF Format 2.0
=====================
Domain ids are now 7 characters long to accommodate chains with more than 9 domains.

There is a new sequence family level at 60% sequence identity (column 7)

A count for each domain in a S100 sequence family is included (column 10) so that it is possible to represent domains uniquely using the CATH code.

C.A.T.H.S.O.L.I.D

C - Class
A - Architecture
T - Topology
H - Homologous Superfamily
S - Sequence Family (S35)
O - Orthogous Seqeuce Family (S60)
L - 'Like' Sequence Family (S95)
I - Identical (S100)
D - Domain (S100 count)

Clustering for the CLF uses a directed multi-linkage clustering algorithm and is order by increasing resolution, domain length and domain_id.
