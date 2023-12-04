#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from scipy.sparse import csr_matrix

# Gene sets used for manifold-learning

# Satija et al. 2015, Nature Biotechnology: https://doi.org/10.1038/nbt.3192
SMALL_CYCLING_GENE_SET = np.array(['Anln', 'Anp32e', 'Atad2', 'Aurka', 'Aurkb', 'Birc5', 'Blm',
       'Brip1', 'Bub1', 'Casp8ap2', 'Cbx5', 'Ccnb2', 'Ccne2', 'Cdc20',
       'Cdc25c', 'Cdc45', 'Cdc6', 'Cdca2', 'Cdca3', 'Cdca7', 'Cdca8',
       'Cdk1', 'Cenpa', 'Cenpe', 'Cenpf', 'Cenpu', 'Chaf1b', 'Ckap2',
       'Ckap2l', 'Ckap5', 'Cks1b', 'Cks2', 'Clspn', 'Ctcf', 'Dlgap5',
       'Dscc1', 'Dtl', 'E2f8', 'Ect2', 'Esco2', 'Exo1', 'Fen1', 'G2e3',
       'Gas2l3', 'Gins2', 'Gmnn', 'Gtse1', 'Hells', 'Hjurp', 'Hmgb2',
       'Hmmr', 'Jpt1', 'Kif11', 'Kif20b', 'Kif23', 'Kif2c', 'Lbr', 'Mcm2',
       'Mcm4', 'Mcm5', 'Mcm6', 'Mki67', 'Msh2', 'Nasp', 'Ncapd2', 'Ndc80',
       'Nek2', 'Nuf2', 'Nusap1', 'Pcna', 'Pimreg', 'Pola1', 'Pold3',
       'Prim1', 'Psrc1', 'Rad51', 'Rad51ap1', 'Rangap1', 'Rfc2', 'Rpa2',
       'Rrm1', 'Rrm2', 'Slbp', 'Smc4', 'Tacc3', 'Tipin', 'Tmpo', 'Top2a',
       'Tpx2', 'Ttk', 'Tubb4b', 'Tyms', 'Ube2c', 'Ubr7', 'Uhrf1', 'Ung',
       'Usp1', 'Wdr76'])

# Riba et al. 2022, Nature Communications: https://doi.org/10.1038/s41467-022-30545-8
MEDIUM_CYCLING_GENE_SET = np.array(['Ankrd17', 'Anln', 'Anp32b', 'Anp32e', 'Apbb2', 'Arl6ip1', 'Aspm',
       'Atad2', 'Atrx', 'Aurka', 'Aurkb', 'Azin1', 'Birc5', 'Blm', 'Bora',
       'Brca2', 'Brd4', 'Brip1', 'Bub1', 'Bub1b', 'Bub3', 'Calm2', 'Calr',
       'Casp8ap2', 'Cbx5', 'Ccna2', 'Ccnb1', 'Ccnb2', 'Ccnd1', 'Ccne1',
       'Ccne2', 'Ccnf', 'Cdc20', 'Cdc25a', 'Cdc25c', 'Cdc26', 'Cdc27',
       'Cdc45', 'Cdc6', 'Cdca2', 'Cdca3', 'Cdca5', 'Cdca7', 'Cdca8',
       'Cdk1', 'Cdk5rap2', 'Cdk7', 'Cdk9', 'Cdkn1b', 'Cdkn2d', 'Cdkn3',
       'Cdt1', 'Cenpa', 'Cenpe', 'Cenpf', 'Cenpu', 'Cep120', 'Cep192',
       'Cep85', 'Chaf1b', 'Chek2', 'Chmp5', 'Chordc1', 'Cit', 'Ckap2',
       'Ckap2l', 'Ckap5', 'Cks1b', 'Cks2', 'Clspn', 'Cradd', 'Crebbp',
       'Crlf3', 'Ctcf', 'Dbf4', 'Dctn1', 'Ddx11', 'Dlgap5', 'Dot1l',
       'Dscc1', 'Dtl', 'Dync1li1', 'Dyrk3', 'E2f1', 'E2f3', 'E2f8',
       'Ect2', 'Esco2', 'Exo1', 'Ezh2', 'Fam83d', 'Fbxo5', 'Fen1', 'Fzr1',
       'G2e3', 'Gadd45b', 'Gas2l3', 'Gigyf2', 'Gins2', 'Gmnn', 'Gtse1',
       'Hat1', 'Hells', 'Hjurp', 'Hmgb2', 'Hmmr', 'Hsp90ab1', 'Hspa8',
       'Incenp', 'Ino80', 'Jade1', 'Jan1', 'Jpt1', 'Junb', 'Kif11',
       'Kif14', 'Kif20a', 'Kif20b', 'Kif23', 'Kif2c', 'Kif4', 'Kifc1',
       'Lbr', 'Mad1l1', 'Mad2l1', 'Mastl', 'Mcm2', 'Mcm4', 'Mcm5', 'Mcm6',
       'Mcph1', 'Mepce', 'Mis18bp1', 'Mki67', 'Msh2', 'Nanog', 'Nasp',
       'Ncapd2', 'Ndc80', 'Nek2', 'Nipbl', 'Nuf2', 'Numa1', 'Nusap1',
       'Orc1', 'Pcna', 'Phb2', 'Phip', 'Pik3c3', 'Pimreg', 'Pin1', 'Pkp4',
       'Plk1', 'Pola1', 'Pold3', 'Ppp2ca,', 'Prc1', 'Prim1', 'Psrc1',
       'Pttg1', 'Pum1', 'Racgap1', 'Rad21', 'Rad50', 'Rad51', 'Rad51ap1',
       'Ranbp1', 'Rangap1', 'Rbm38', 'Rcc1', 'Rdx', 'Rfc2', 'Rhoa',
       'Riok2', 'Rnf167', 'Rnf4', 'Rpa2', 'Rpa3', 'Rptor', 'Rrm1', 'Rrm2',
       'Sde2', 'Senp6', 'Sfpq', 'Sgo2a', 'Slbp', 'Smc4', 'Son', 'Spag5',
       'Spdl1', 'Srpk2', 'Tacc3', 'Taf6', 'Taok3', 'Tfdp2', 'Ticrr',
       'Timeless', 'Tipin', 'Tmpo', 'Top2a', 'Topbp1', 'Tpx2', 'Trim59',
       'Ttc28', 'Ttk', 'Tuba1c', 'Tubb4b', 'Tyms', 'Ube2c', 'Ubr7',
       'Uhrf1', 'Ung', 'Usp1', 'Wdr76', 'Wee1', 'Ythdf2', 'Zfp36l1',
       'Zwint'])

# The Gene Ontology Consortium. 2023, Genetics: https://doi.org/10.1093/genetics/iyad031
# Gene ontology terms "cell cycle" and "cell division"
LARGE_CYCLING_GENE_SET = np.array(['1700013H16Rik', '1700028K03Rik', '1700040F15Rik', '2610528A11Rik',
       '3830403N18Rik', '4930447C04Rik', '4933427D14Rik', 'AY074887',
       'Aaas', 'Aatf', 'Abcb1a', 'Abcb1b', 'Abl1', 'Abraxas1', 'Abraxas2',
       'Actb', 'Actl6a', 'Actl6b', 'Actr2', 'Actr3', 'Actr5', 'Actr8',
       'Acvr1', 'Acvr1b', 'Adam17', 'Adamts1', 'Adarb1', 'Adcyap1',
       'Afap1l2', 'Ago4', 'Ahctf1', 'Ahr', 'Aicda', 'Aif1', 'Ajuba',
       'Ak1', 'Akap8', 'Akap8l', 'Akna', 'Akt1', 'Alkbh4', 'Alms1',
       'Alox8', 'Ambra1', 'Anapc1', 'Anapc10', 'Anapc11', 'Anapc13',
       'Anapc15', 'Anapc16', 'Anapc2', 'Anapc4', 'Anapc5', 'Anapc7',
       'Angel2', 'Ank3', 'Ankfn1', 'Ankk1', 'Ankle1', 'Ankle2', 'Ankrd17',
       'Ankrd31', 'Ankrd53', 'Anln', 'Anp32b', 'Anp32e', 'Anxa1',
       'Anxa11', 'Apbb1', 'Apbb2', 'Apbb3', 'Apc', 'Apex1', 'Apex2',
       'App', 'Appl1', 'Appl2', 'Arf1', 'Arf6', 'Arhgef10', 'Arhgef2',
       'Arid1a', 'Arid1b', 'Arid2', 'Arl2', 'Arl3', 'Arl6ip1', 'Arl8a',
       'Arl8b', 'Arntl', 'Arpp19', 'Ascl1', 'Asns', 'Aspm', 'Asz1',
       'Atad2', 'Atad5', 'Atf2', 'Atf5', 'Atm', 'Atp2b4', 'Atr', 'Atrip',
       'Atrx', 'Aunip', 'Aurka', 'Aurkb', 'Aurkc', 'Aven', 'Avpi1',
       'Axin2', 'Azi2', 'Azin1', 'BC004004', 'BC005624', 'BC034090',
       'Babam1', 'Babam2', 'Bach1', 'Bag6', 'Bak1', 'Banf1', 'Banp',
       'Bap1', 'Bard1', 'Bax', 'Baz1b', 'Bbs4', 'Bccip', 'Bcl2', 'Bcl2l1',
       'Bcl2l11', 'Bcl7a', 'Bcl7b', 'Bcl7c', 'Bcr', 'Becn1', 'Bex2',
       'Bex4', 'Bid', 'Bin1', 'Bin3', 'Birc2', 'Birc3', 'Birc5', 'Birc6',
       'Birc7', 'Blcap', 'Blm', 'Bmi1', 'Bmp2', 'Bmp4', 'Bmp7', 'Bmyc',
       'Bnip2', 'Bod1', 'Boll', 'Bop1', 'Bora', 'Brca1', 'Brca2', 'Brcc3',
       'Brcc3dc', 'Brd4', 'Brd7', 'Brd8', 'Brdt', 'Brinp1', 'Brinp2',
       'Brinp3', 'Brip1', 'Brme1', 'Brsk1', 'Brsk2', 'Btbd18', 'Btc',
       'Btg1', 'Btg1b', 'Btg1c', 'Btg2', 'Btg3', 'Btg4', 'Btn2a2', 'Btrc',
       'Bub1', 'Bub1b', 'Bub3', 'C2cd3', 'Cables1', 'Cables2', 'Cacnb4',
       'Cacul1', 'Calm1', 'Calm2', 'Calm3', 'Calr', 'Camk1', 'Camk2a',
       'Camk2b', 'Camk2d', 'Camk2g', 'Capn3', 'Casp2', 'Casp3',
       'Casp8ap2', 'Cast', 'Cat', 'Catsperz', 'Cbx5', 'Ccar1', 'Ccar2',
       'Ccdc124', 'Ccdc57', 'Ccdc61', 'Ccdc69', 'Ccdc8', 'Ccdc84',
       'Ccl12', 'Ccl2', 'Ccn2', 'Ccna1', 'Ccna2', 'Ccnb1', 'Ccnb1ip1',
       'Ccnb2', 'Ccnb3', 'Ccnc', 'Ccnd1', 'Ccnd2', 'Ccnd3', 'Ccndbp1',
       'Ccne1', 'Ccne2', 'Ccnf', 'Ccng1', 'Ccng2', 'Ccnh', 'Ccni', 'Ccnj',
       'Ccnjl', 'Ccnk', 'Ccnl1', 'Ccnl2', 'Ccno', 'Ccnq', 'Ccnt1',
       'Ccnt2', 'Ccny', 'Ccp110', 'Ccpg1', 'Ccsap', 'Cd28', 'Cd2ap',
       'Cdc123', 'Cdc14a', 'Cdc14b', 'Cdc16', 'Cdc20', 'Cdc23', 'Cdc25a',
       'Cdc25b', 'Cdc25c', 'Cdc26', 'Cdc27', 'Cdc34', 'Cdc42', 'Cdc45',
       'Cdc5l', 'Cdc6', 'Cdc7', 'Cdc73', 'Cdca2', 'Cdca3', 'Cdca5',
       'Cdca7', 'Cdca8', 'Cdk1', 'Cdk10', 'Cdk11b', 'Cdk14', 'Cdk15',
       'Cdk16', 'Cdk17', 'Cdk18', 'Cdk2', 'Cdk20', 'Cdk2ap2', 'Cdk3',
       'Cdk4', 'Cdk5', 'Cdk5r1', 'Cdk5rap1', 'Cdk5rap2', 'Cdk5rap3',
       'Cdk6', 'Cdk7', 'Cdk9', 'Cdkl1', 'Cdkn1a', 'Cdkn1b', 'Cdkn1c',
       'Cdkn2a', 'Cdkn2b', 'Cdkn2c', 'Cdkn2d', 'Cdkn3', 'Cdt1', 'Cebpa',
       'Celf1', 'Cenpa', 'Cenpc1', 'Cenpe', 'Cenpf', 'Cenph', 'Cenpj',
       'Cenpk', 'Cenpq', 'Cenps', 'Cenpt', 'Cenpu', 'Cenpv', 'Cenpw',
       'Cenpx', 'Cep120', 'Cep126', 'Cep131', 'Cep135', 'Cep152',
       'Cep164', 'Cep192', 'Cep250', 'Cep295', 'Cep295nl', 'Cep44',
       'Cep55', 'Cep63', 'Cep68', 'Cep72', 'Cep76', 'Cep85', 'Cep97',
       'Cetn1', 'Cetn2', 'Cetn3', 'Cetn4', 'Cfl1', 'Cgref1', 'Cgrrf1',
       'Chaf1a', 'Chaf1b', 'Champ1', 'Chd3', 'Chek1', 'Chek2', 'Chfr',
       'Chmp1a', 'Chmp1b', 'Chmp1b2', 'Chmp2a', 'Chmp2b', 'Chmp3',
       'Chmp4b', 'Chmp4c', 'Chmp5', 'Chmp6', 'Chmp7', 'Chordc1', 'Chtf18',
       'Cib1', 'Cinp', 'Cirbp', 'Cit', 'Cited2', 'Ckap2', 'Ckap2l',
       'Ckap5', 'Cks1b', 'Cks1brt', 'Cks2', 'Clasp1', 'Clasp2', 'Clgn',
       'Clic1', 'Clock', 'Clspn', 'Clta', 'Cltc', 'Cnppd1', 'Cntd1',
       'Cntln', 'Cntrl', 'Cntrob', 'Commd5', 'Cops5', 'Cpeb1', 'Cpsf3',
       'Cradd', 'Crebbp', 'Crlf3', 'Crnn', 'Crocc', 'Cry1', 'Csf1r',
       'Csnk1a1', 'Csnk1d', 'Csnk2a1', 'Csnk2a2', 'Cspp1', 'Ctbp1',
       'Ctc1', 'Ctcf', 'Ctdp1', 'Ctdsp1', 'Ctdsp2', 'Ctdspl', 'Ctnnb1',
       'Cts7', 'Cul3', 'Cul4a', 'Cul4b', 'Cul7', 'Cul9', 'Cuzd1', 'Cxcr5',
       'Cyld', 'Cyp1a1', 'Cyp26b1', 'Cyp27b1', 'D1Pas1', 'D7Ertd443e',
       'Dab2ip', 'Dach1', 'Dact1', 'Dapk3', 'Daxx', 'Dazl', 'Dbf4',
       'Dclre1a', 'Dct', 'Dctn1', 'Dctn2', 'Dctn3', 'Dctn6', 'Dcun1d3',
       'Ddb1', 'Ddias', 'Ddit3', 'Ddr2', 'Ddx11', 'Ddx39b', 'Ddx3x',
       'Ddx4', 'Deup1', 'Dgkz', 'Dicer1', 'Dis3l2', 'Dixdc1', 'Dlg1',
       'Dlgap5', 'Dll1', 'Dmap1', 'Dmc1', 'Dmd', 'Dmrt1', 'Dmrtc2',
       'Dmtf1', 'Dna2', 'Dnmt3c', 'Dnmt3l', 'Dock7', 'Donson', 'Dot1l',
       'Dpf1', 'Dpf2', 'Dpf3', 'Dppa3', 'Dr1', 'Drd2', 'Drd3', 'Drg1',
       'Dscc1', 'Dsn1', 'Dstn', 'Dtl', 'Dtx3l', 'Dusp1', 'Dusp3',
       'Dync1h1', 'Dync1li1', 'Dynlt1b', 'Dynlt3', 'Dyrk3', 'E2f1',
       'E2f2', 'E2f3', 'E2f4', 'E2f5', 'E2f6', 'E2f7', 'E2f8', 'E4f1',
       'Ecd', 'Ecrg4', 'Ect2', 'Edn1', 'Edn3', 'Ednra', 'Eef1aknmt',
       'Efhc1', 'Efhc2', 'Egf', 'Egfr', 'Ehmt2', 'Eid1', 'Eif2ak4',
       'Eif4e', 'Eif4ebp1', 'Eif4g1', 'Eif4g3', 'Eme1', 'Eme2', 'Eml1',
       'Eml3', 'Eml4', 'Enkd1', 'Ensa', 'Entr1', 'Ep300', 'Ep400',
       'Epb41', 'Epb41l2', 'Epc1', 'Epc2', 'Epgn', 'Epm2a', 'Eps8',
       'Ercc1', 'Ercc2', 'Ercc3', 'Ercc4', 'Ercc6', 'Ercc6l', 'Ereg',
       'Esco1', 'Esco2', 'Espl1', 'Esr1', 'Esrrb', 'Esx1', 'Etaa1',
       'Ets1', 'Etv5', 'Evi2b', 'Evi5', 'Exd1', 'Exo1', 'Exoc1', 'Exoc2',
       'Exoc3', 'Exoc4', 'Exoc5', 'Exoc6', 'Exoc6b', 'Exoc7', 'Exoc8',
       'Ext1', 'Eya1', 'Ezh2', 'Ezr', 'Fam107a', 'Fam110a', 'Fam122a',
       'Fam122c', 'Fam32a', 'Fam83d', 'Fanca', 'Fancd2', 'Fanci', 'Fancm',
       'Fap', 'Fbxl12', 'Fbxl15', 'Fbxl17', 'Fbxl21', 'Fbxl22', 'Fbxl3',
       'Fbxl6', 'Fbxl7', 'Fbxl8', 'Fbxo31', 'Fbxo4', 'Fbxo43', 'Fbxo5',
       'Fbxo7', 'Fbxw11', 'Fbxw5', 'Fbxw7', 'Fem1b', 'Fen1', 'Fes',
       'Fgf1', 'Fgf10', 'Fgf13', 'Fgf2', 'Fgf3', 'Fgf4', 'Fgf5', 'Fgf6',
       'Fgf7', 'Fgf8', 'Fgf9', 'Fgfr1', 'Fgfr2', 'Fgfr3', 'Fhl1', 'Fign',
       'Fignl1', 'Fkbp6', 'Flcn', 'Flna', 'Flt3l', 'Fmn2', 'Fnta', 'Fntb',
       'Fosl1', 'Foxa1', 'Foxc1', 'Foxe3', 'Foxg1', 'Foxj2', 'Foxj3',
       'Foxk1', 'Foxm1', 'Foxn3', 'Foxo4', 'Fsd1', 'Fubp1', 'Fut10',
       'Fzd3', 'Fzd7', 'Fzd9', 'Fzr1', 'G2e3', 'Gadd45a', 'Gadd45b',
       'Gadd45g', 'Gadd45gip1', 'Gak', 'Garem1', 'Gas1', 'Gas2', 'Gas2l1',
       'Gas2l3', 'Gata3', 'Gata4', 'Gata6', 'Gbf1', 'Gcna', 'Gdpd5',
       'Gem', 'Gen1', 'Gigyf2', 'Gins1', 'Gins2', 'Gins3', 'Gipc1',
       'Git1', 'Gja1', 'Gjc2', 'Gkn1', 'Gli1', 'Gm10230', 'Gm10488',
       'Gm1140', 'Gm14525', 'Gm16430', 'Gm1993', 'Gm2012', 'Gm2030',
       'Gm20736', 'Gm20817', 'Gm20820', 'Gm20824', 'Gm20843', 'Gm20890',
       'Gm20911', 'Gm21095', 'Gm21117', 'Gm21294', 'Gm21627', 'Gm21760',
       'Gm21858', 'Gm21865', 'Gm21996', 'Gm28102', 'Gm28490', 'Gm28510',
       'Gm28576', 'Gm28870', 'Gm28919', 'Gm28961', 'Gm29276', 'Gm29554',
       'Gm29866', 'Gm30731', 'Gm4297', 'Gm49340', 'Gm49361', 'Gm49601',
       'Gm5168', 'Gm5169', 'Gm5934', 'Gm5935', 'Gm6121', 'Gm773', 'Gm960',
       'Gmnc', 'Gmnn', 'Gnai1', 'Gnai2', 'Gnai3', 'Gnl3', 'Golga2',
       'Gper1', 'Gpnmb', 'Gpr132', 'Gpr3', 'Gpsm1', 'Gpsm2', 'Grb14',
       'Grk5', 'Gsk3b', 'Gspt2', 'Gtf2b', 'Gtpbp4', 'Gtse1', 'H1f8',
       'H2ax', 'Hacd1', 'Hace1', 'Haspin', 'Hat1', 'Haus1', 'Haus2',
       'Haus3', 'Haus4', 'Haus5', 'Haus6', 'Haus7', 'Haus8', 'Hcfc1',
       'Hdac3', 'Hdac8', 'Heca', 'Hecw2', 'Hells', 'Hepacam', 'Hepacam2',
       'Hes1', 'Hexim1', 'Hexim2', 'Hfm1', 'Hhex', 'Hinfp', 'Hjurp',
       'Hmcn1', 'Hmg20b', 'Hmga2', 'Hmgb1', 'Hmgb2', 'Hmmr', 'Hnf4a',
       'Hnrnpu', 'Hormad1', 'Hormad2', 'Hoxa13', 'Hoxb4', 'Hpgd', 'Hras',
       'Hsf1', 'Hsf2bp', 'Hsp90ab1', 'Hspa1a', 'Hspa1b', 'Hspa2', 'Hspa8',
       'Htr2b', 'Htt', 'Hus1', 'Hus1b', 'Hyal1', 'Id2', 'Id3', 'Id4',
       'Ier3', 'Iffo1', 'Ifnz', 'Igf1', 'Igf1r', 'Igf2', 'Iho1', 'Ik',
       'Ikzf1', 'Il10', 'Il1a', 'Il1b', 'Ilk', 'Ilkap', 'Inca1', 'Incenp',
       'Ing1', 'Ing2', 'Ing3', 'Ing4', 'Ing5', 'Inha', 'Inhba', 'Inip',
       'Ino80', 'Ino80b', 'Ino80c', 'Ino80d', 'Ino80e', 'Ins1', 'Ins2',
       'Insc', 'Insm1', 'Insm2', 'Insr', 'Ints13', 'Ints3', 'Ints7',
       'Intu', 'Iqgap1', 'Iqgap3', 'Irf1', 'Ist1', 'Itgb1', 'Itgb1bp1',
       'Itgb3bp', 'Jade1', 'Jade2', 'Jade3', 'Jan1', 'Jpt1', 'Jtb', 'Jun',
       'Junb', 'Jund', 'Kank2', 'Kash5', 'Kat14', 'Kat2a', 'Kat2b',
       'Kat5', 'Kat7', 'Katna1', 'Katnb1', 'Kcna5', 'Kcnh5', 'Kctd11',
       'Kdf1', 'Kdm8', 'Khdc3', 'Khdrbs1', 'Kif11', 'Kif13a', 'Kif14',
       'Kif15', 'Kif18a', 'Kif18b', 'Kif20a', 'Kif20b', 'Kif22', 'Kif23',
       'Kif2a', 'Kif2b', 'Kif2c', 'Kif3a', 'Kif3b', 'Kif4', 'Kifc1',
       'Kifc2', 'Kifc5b', 'Kit', 'Kiz', 'Klf11', 'Klf4', 'Klhdc3',
       'Klhdc8b', 'Klhl13', 'Klhl18', 'Klhl21', 'Klhl22', 'Klhl42',
       'Klhl9', 'Kmt2e', 'Kmt5a', 'Knl1', 'Knstrn', 'Kntc1', 'Kpnb1',
       'Krtap21-1', 'L3mbtl1', 'Larp7', 'Lats1', 'Lats2', 'Lbh', 'Lbr',
       'Lcmt1', 'Lef1', 'Lemd3', 'Lep', 'Lfng', 'Lgmn', 'Lif', 'Lig1',
       'Lig3', 'Lig4', 'Limk2', 'Lin54', 'Lin9', 'Llgl1', 'Llgl2', 'Lmln',
       'Lmna', 'Lmnb1', 'Lrp6', 'Lrrcc1', 'Lsm10', 'Lsm11', 'Lsm14a',
       'Lzts1', 'Lzts2', 'M1ap', 'Macroh2a1', 'Mad1l1', 'Mad2l1',
       'Mad2l1bp', 'Mad2l2', 'Madd', 'Maea', 'Mael', 'Majin', 'Map10',
       'Map1s', 'Map2k1', 'Map3k11', 'Map3k20', 'Map3k8', 'Map4', 'Map9',
       'Mapk1', 'Mapk12', 'Mapk13', 'Mapk14', 'Mapk15', 'Mapk1ip1',
       'Mapk3', 'Mapk4', 'Mapk6', 'Mapk7', 'Mapk8', 'Mapre1', 'Mapre2',
       'Mapre3', 'Marf1', 'Mark4', 'Marveld1', 'Mastl', 'Mau2', 'Mbd4',
       'Mbip', 'Mblac1', 'Mbtd1', 'Mcidas', 'Mcm2', 'Mcm3', 'Mcm4',
       'Mcm5', 'Mcm6', 'Mcm7', 'Mcm8', 'Mcmbp', 'Mcmdc2', 'Mcph1',
       'Mcrs1', 'Mcts1', 'Mdc1', 'Mdk', 'Mdm1', 'Mdm2', 'Mdm4', 'Meaf6',
       'Mecom', 'Mecp2', 'Med1', 'Mei1', 'Mei4', 'Meig1', 'Meikin',
       'Meiob', 'Meioc', 'Meiosin', 'Meis2', 'Melk', 'Men1', 'Mepce',
       'Met', 'Mettl3', 'Mfn2', 'Mical3', 'Miip', 'Mir1186', 'Mir124a-1',
       'Mir124a-2', 'Mir124a-3', 'Mir16-1', 'Mir214', 'Mir26a-1',
       'Mir26a-2', 'Mir26b', 'Mir664', 'Mir744', 'Mis12', 'Mis18a',
       'Mis18bp1', 'Misp', 'Mitd1', 'Mki67', 'Mlf1', 'Mlh1', 'Mlh3',
       'Mllt3', 'Mn1', 'Mnat1', 'Mnd1', 'Mns1', 'Mnt', 'Morc2b',
       'Morf4l1', 'Morf4l2', 'Mos', 'Mov10l1', 'Mpl', 'Mplkip', 'Mre11a',
       'Mrgbp', 'Mrnip', 'Mrpl41', 'Ms4a3', 'Msh2', 'Msh4', 'Msh5',
       'Msx1', 'Msx2', 'Mta3', 'Mtbp', 'Mtus1', 'Muc1', 'Mus81', 'Myb',
       'Mybbp1a', 'Mybl1', 'Mybl2', 'Myc', 'Myh10', 'Myh9', 'Mylk2',
       'Myo16', 'Myo19', 'Myocd', 'Myog', 'Mzt1', 'Naa10', 'Naa50',
       'Nabp1', 'Nabp2', 'Nacc2', 'Nae1', 'Nanog', 'Nanos2', 'Nanos3',
       'Nap1l2', 'Nasp', 'Nat10', 'Nbn', 'Ncapd2', 'Ncapd3', 'Ncapg',
       'Ncapg2', 'Ncaph', 'Ncaph2', 'Ncoa3', 'Ncor1', 'Ndc1', 'Ndc80',
       'Nde1', 'Ndel1', 'Nedd1', 'Nedd9', 'Nek1', 'Nek10', 'Nek11',
       'Nek2', 'Nek3', 'Nek4', 'Nek6', 'Nek9', 'Nes', 'Neurog1', 'Nf2',
       'Nfatc1', 'Nfe2l1', 'Nfrkb', 'Nin', 'Nipbl', 'Nkx3-1', 'Nle1',
       'Nlrp5', 'Nme6', 'Nop53', 'Notch1', 'Npat', 'Npm1', 'Npm2', 'Nppc',
       'Npr2', 'Nr2c2', 'Nr2e1', 'Nr2f2', 'Nr4a1', 'Nr4a3', 'Nras',
       'Nrde2', 'Nsfl1c', 'Nsl1', 'Nsmce2', 'Nsun2', 'Ntmt1', 'Nubp1',
       'Nudc', 'Nudt15', 'Nudt16', 'Nudt6', 'Nuf2', 'Nuggc', 'Numa1',
       'Numb', 'Numbl', 'Nup153', 'Nup214', 'Nup37', 'Nup43', 'Nup62',
       'Nup88', 'Nupr1', 'Nupr1l', 'Nusap1', 'Obox4', 'Obsl1', 'Odf2',
       'Ofd1', 'Oip5', 'Ooep', 'Orc1', 'Orc4', 'Orc6', 'Osm', 'Ovol1',
       'Ovol2', 'Padi6', 'Paf1', 'Pafah1b1', 'Pagr1a', 'Pagr1b', 'Pak4',
       'Pard3', 'Pard3b', 'Pard6a', 'Pard6b', 'Pard6g', 'Parp3', 'Parp9',
       'Pax6', 'Paxip1', 'Pbk', 'Pbrm1', 'Pbx1', 'Pcid2', 'Pclaf', 'Pcm1',
       'Pcna', 'Pcnp', 'Pcnt', 'Pdcd2l', 'Pdcd6ip', 'Pde3a', 'Pde4dip',
       'Pdgfa', 'Pdgfb', 'Pdgfc', 'Pdgfd', 'Pdgfrb', 'Pdik1l', 'Pdpn',
       'Pds5a', 'Pds5b', 'Pdxp', 'Pebp1', 'Pelo', 'Per2', 'Pes1', 'Pgf',
       'Pggt1b', 'Phactr4', 'Phb2', 'Phf10', 'Phf13', 'Phf8', 'Phgdh',
       'Phip', 'Pias1', 'Pibf1', 'Pidd1', 'Pik3c3', 'Pik3cb', 'Pik3r4',
       'Pim1', 'Pim2', 'Pim3', 'Pimreg', 'Pin1', 'Pinx1', 'Piwil1',
       'Piwil2', 'Piwil4', 'Pkd1', 'Pkd2', 'Pkhd1', 'Pkia', 'Pkmyt1',
       'Pkn2', 'Pkp4', 'Plaat3', 'Plcb1', 'Plcg2', 'Pld6', 'Plec', 'Plk1',
       'Plk2', 'Plk3', 'Plk4', 'Plk5', 'Plpp2', 'Plrg1', 'Plscr1',
       'Plscr2', 'Pmf1', 'Pml', 'Pmp22', 'Pms2', 'Pnpt1', 'Poc1a',
       'Poc1b', 'Poc5', 'Pogz', 'Pola1', 'Pold3', 'Poldip2', 'Pole',
       'Pou3f2', 'Pou3f3', 'Pou4f1', 'Pou5f1', 'Ppm1d', 'Ppm1g', 'Ppp1ca',
       'Ppp1cb', 'Ppp1cc', 'Ppp1r10', 'Ppp1r12a', 'Ppp1r13b', 'Ppp1r1c',
       'Ppp1r35', 'Ppp2ca', 'Ppp2ca,', 'Ppp2cb', 'Ppp2r1a', 'Ppp2r2d',
       'Ppp2r3d', 'Ppp2r5b', 'Ppp2r5c', 'Ppp3ca', 'Ppp6c', 'Prap1',
       'Prc1', 'Prcc', 'Prdm11', 'Prdm15', 'Prdm5', 'Prdm9', 'Prickle1',
       'Prim1', 'Prkaca', 'Prkacb', 'Prkca', 'Prkcd', 'Prkce', 'Prkcq',
       'Prkdc', 'Prmt2', 'Prok1', 'Prox1', 'Prpf19', 'Prpf40a', 'Prr11',
       'Prr19', 'Prr5', 'Psma8', 'Psmc3ip', 'Psmd10', 'Psmd13', 'Psme1',
       'Psme2', 'Psme3', 'Psmg2', 'Psrc1', 'Pstpip1', 'Ptch1', 'Pten',
       'Ptgs2', 'Ptn', 'Ptp4a1', 'Ptpa', 'Ptpn11', 'Ptpn3', 'Ptpn6',
       'Ptprc', 'Ptprk', 'Ptprv', 'Pttg1', 'Pum1', 'Rab10', 'Rab11a',
       'Rab11fip3', 'Rab11fip4', 'Rab35', 'Rabgap1', 'Racgap1', 'Rack1',
       'Rad1', 'Rad17', 'Rad21', 'Rad21l', 'Rad23a', 'Rad50', 'Rad51',
       'Rad51ap1', 'Rad51b', 'Rad51c', 'Rad51d', 'Rad54b', 'Rad54l',
       'Rad9a', 'Rad9b', 'Rae1', 'Rala', 'Ralb', 'Ran', 'Ranbp1',
       'Rangap1', 'Rara', 'Rassf1', 'Rassf2', 'Rassf4', 'Rb1', 'Rb1cc1',
       'Rbbp4', 'Rbbp8', 'Rbl1', 'Rbl2', 'Rbm38', 'Rbm7', 'Rcbtb1',
       'Rcc1', 'Rcc2', 'Rdx', 'Rec114', 'Rec8', 'Recql4', 'Recql5',
       'Reep3', 'Reep4', 'Rfc2', 'Rfwd3', 'Rgcc', 'Rgs14', 'Rgs2',
       'Rhno1', 'Rhoa', 'Rhob', 'Rhoc', 'Rhou', 'Rif1', 'Rint1', 'Riok2',
       'Ripor2', 'Rmi1', 'Rmi2', 'Rnaseh2b', 'Rnf112', 'Rnf167', 'Rnf2',
       'Rnf212', 'Rnf212b', 'Rnf4', 'Rnf8', 'Rny1', 'Rny3', 'Rock1',
       'Rock2', 'Rpa1', 'Rpa2', 'Rpa3', 'Rpl10l', 'Rpl17', 'Rpl23',
       'Rpl24', 'Rpl26', 'Rprd1b', 'Rprm', 'Rps15a', 'Rps27l', 'Rps3',
       'Rps6', 'Rps6ka2', 'Rps6ka3', 'Rps6kb1', 'Rptor', 'Rrm1', 'Rrm2',
       'Rrm2b', 'Rrp8', 'Rrs1', 'Rsph1', 'Rspo1', 'Rtel1', 'Rtf2', 'Rtkn',
       'Rttn', 'Runx3', 'Ruvbl1', 'Ruvbl2', 'Rxfp3', 'Sac3d1', 'Samd9l',
       'Sapcd2', 'Sass6', 'Sbds', 'Scrib', 'Sdcbp', 'Sdccag8', 'Sde2',
       'Seh1l', 'Senp2', 'Senp5', 'Senp6', 'Septin1', 'Septin10',
       'Septin11', 'Septin12', 'Septin14', 'Septin2', 'Septin3',
       'Septin4', 'Septin5', 'Septin6', 'Septin7', 'Septin8', 'Septin9',
       'Setd2', 'Setdb2', 'Setmar', 'Sfn', 'Sfpq', 'Sfrp1', 'Sfrp2',
       'Sgf29', 'Sgk1', 'Sgo1', 'Sgo2a', 'Sgo2b', 'Sgsm3', 'Sh2b1',
       'Sh3glb1', 'Shb', 'Shcbp1l', 'Shh', 'Shoc1', 'Siah1a', 'Siah2',
       'Sik1', 'Sin3a', 'Sin3b', 'Sipa1', 'Sirt1', 'Sirt2', 'Sirt7',
       'Six3', 'Ska1', 'Ska2', 'Ska3', 'Skil', 'Skp2', 'Slbp', 'Slc16a1',
       'Slc25a31', 'Slc26a8', 'Slc6a4', 'Slc9a3r1', 'Slf1', 'Slf2',
       'Slfn1', 'Slx', 'Slx4', 'Slxl1', 'Smarca2', 'Smarca4', 'Smarca5',
       'Smarcad1', 'Smarcb1', 'Smarcc1', 'Smarcc2', 'Smarcd1', 'Smarcd2',
       'Smarcd3', 'Smarce1', 'Smc1a', 'Smc1b', 'Smc2', 'Smc3', 'Smc4',
       'Smc5', 'Smim22', 'Smoc2', 'Smpd3', 'Smyd5', 'Snd1', 'Snx18',
       'Snx33', 'Snx9', 'Son', 'Sox15', 'Sox17', 'Sox2', 'Sox5', 'Sox9',
       'Spag5', 'Spag6l', 'Spag8', 'Spast', 'Spata22', 'Spc24', 'Spc25',
       'Spdl1', 'Spdya', 'Spdye4a', 'Specc1l', 'Spg20', 'Sphk1', 'Spice1',
       'Spin1', 'Spin2c', 'Spire1', 'Spire2', 'Spo11', 'Spout1', 'Spry1',
       'Spry2', 'Sptbn1', 'Sra1', 'Src', 'Srpk2', 'Ssna1', 'Sstr5',
       'Ssx2ip', 'Stag1', 'Stag2', 'Stag3', 'Stambp', 'Stard13', 'Stard9',
       'Stat3', 'Stat5a', 'Stat5b', 'Steap3', 'Stil', 'Stk10', 'Stk11',
       'Stk33', 'Stk35', 'Stmn1', 'Stox1', 'Stra8', 'Strada', 'Stradb',
       'Stx2', 'Stxbp4', 'Sun1', 'Sun2', 'Susd2', 'Suv39h1', 'Suv39h2',
       'Syce1', 'Syce1l', 'Syce2', 'Syce3', 'Sycp1', 'Sycp2', 'Sycp2l',
       'Sycp3', 'Syde1', 'Syf2', 'Tacc1', 'Tacc2', 'Tacc3', 'Tada2a',
       'Tada3', 'Taf1', 'Taf10', 'Taf2', 'Taf6', 'Tafazzin', 'Tal1',
       'Taok1', 'Taok2', 'Taok3', 'Tardbp', 'Tas1r2', 'Tas2r102',
       'Tas2r121', 'Tas2r124', 'Tasor', 'Tbcd', 'Tbce', 'Tbrg1', 'Tbx2',
       'Tbx3', 'Tcf19', 'Tcf3', 'Tcim', 'Tdrd1', 'Tdrd12', 'Tdrd9',
       'Tdrkh', 'Tead3', 'Tent4b', 'Tent5b', 'Terb1', 'Terb2', 'Terf1',
       'Terf2', 'Tert', 'Tesmin', 'Tet2', 'Tex11', 'Tex12', 'Tex14',
       'Tex15', 'Tex19.1', 'Tex19.2', 'Tfap4', 'Tfdp1', 'Tfdp2', 'Tfpt',
       'Tgfa', 'Tgfb1', 'Tgfb2', 'Tgfb3', 'Tgm1', 'Thap1', 'Thbs4',
       'Thoc1', 'Thoc2', 'Thoc5', 'Tial1', 'Ticrr', 'Timeless', 'Timp2',
       'Tipin', 'Tiprl', 'Tjp3', 'Tle6', 'Tlk1', 'Tlk2', 'Tm4sf5',
       'Tmem67', 'Tmigd1', 'Tmod3', 'Tmpo', 'Tmprss11a', 'Tnf', 'Tnfaip3',
       'Tnks', 'Togaram1', 'Togaram2', 'Tom1l1', 'Tom1l2', 'Top1',
       'Top2a', 'Top2b', 'Top3a', 'Topaz1', 'Topbp1', 'Tpd52l1', 'Tppp',
       'Tpr', 'Tpra1', 'Tpx2', 'Trappc12', 'Trex1', 'Trim21', 'Trim32',
       'Trim35', 'Trim36', 'Trim37', 'Trim39', 'Trim59', 'Trim71',
       'Trim75', 'Triobp', 'Trip13', 'Trnp1', 'Trp53', 'Trp53bp1',
       'Trp53bp2', 'Trp53i13', 'Trp63', 'Trp73', 'Trrap', 'Tsc1', 'Tsc2',
       'Tsg101', 'Tspyl2', 'Ttbk1', 'Ttc28', 'Ttk', 'Ttl', 'Ttll12',
       'Ttyh1', 'Tuba1a', 'Tuba1b', 'Tuba1c', 'Tuba3a', 'Tuba4a', 'Tuba8',
       'Tubal3', 'Tubb1', 'Tubb2a', 'Tubb2b', 'Tubb3', 'Tubb4a', 'Tubb4b',
       'Tubb5', 'Tubb6', 'Tubd1', 'Tube1', 'Tubg1', 'Tubg2', 'Tubgcp2',
       'Tubgcp3', 'Tubgcp4', 'Tubgcp5', 'Tubgcp6', 'Tunar', 'Txlng',
       'Txnip', 'Txnl4b', 'Tyms', 'Uba3', 'Ubb', 'Ubd', 'Ube2b', 'Ube2c',
       'Ube2e2', 'Ube2i', 'Ube2l3', 'Ube2s', 'Ubr2', 'Ubr7', 'Ubxn2b',
       'Uchl5', 'Uhmk1', 'Uhrf1', 'Uhrf2', 'Uimc1', 'Ulk4', 'Unc119',
       'Ung', 'Upf1', 'Urgcp', 'Ush1c', 'Usp1', 'Usp16', 'Usp19', 'Usp2',
       'Usp22', 'Usp26', 'Usp28', 'Usp29', 'Usp3', 'Usp33', 'Usp37',
       'Usp39', 'Usp44', 'Usp47', 'Usp51', 'Usp8', 'Usp9x', 'Utp14b',
       'Uvrag', 'Uxt', 'Vangl2', 'Vash1', 'Vcp', 'Vcpip1', 'Vegfa',
       'Vegfb', 'Vegfc', 'Vegfd', 'Vps4a', 'Vps4b', 'Vps72', 'Vrk1',
       'Wac', 'Wapl', 'Washc1', 'Washc5', 'Wasl', 'Wdhd1', 'Wdr12',
       'Wdr5', 'Wdr6', 'Wdr62', 'Wdr76', 'Wee1', 'Wee2', 'Wfs1', 'Wiz',
       'Wnt10b', 'Wnt3a', 'Wnt4', 'Wnt5a', 'Wnt7a', 'Wnt9b', 'Wrap73',
       'Wtap', 'Wwtr1', 'Xiap', 'Xlr', 'Xlr3a', 'Xlr3b', 'Xlr3c', 'Xlr4a',
       'Xlr4b', 'Xlr4c', 'Xlr5a', 'Xlr5b', 'Xlr5c', 'Xpc', 'Xpo1',
       'Xrcc2', 'Xrcc3', 'Xrn1', 'Ybx1', 'Yeats2', 'Yeats4', 'Ythdc2',
       'Ythdf2', 'Ywhae', 'Ywhah', 'Yy1', 'Zbed3', 'Zbed6', 'Zbtb16',
       'Zbtb17', 'Zbtb18', 'Zbtb49', 'Zc3h12d', 'Zc3hc1', 'Zcwpw1',
       'Zfp207', 'Zfp318', 'Zfp365', 'Zfp369', 'Zfp36l1', 'Zfp36l2',
       'Zfp385a', 'Zfp386', 'Zfp449', 'Zfp503', 'Zfp655', 'Zfp703',
       'Zfp830', 'Zfy2', 'Zfyve19', 'Zfyve26', 'Zic1', 'Zic3', 'Zmpste24',
       'Znhit1', 'Zpr1', 'Zscan21', 'Zw10', 'Zwilch', 'Zwint', 'Zzz3'])

# Gene sets used for categorical cell cycle phase inference (G1, S, G2M)
S_genes_mouse = np.array(['Atad2', 'Blm', 'Brip1', 'Casp8ap2', 'Ccne2', 'Cdc45', 'Cdc6',
       'Cdca7', 'Chaf1b', 'Clspn', 'Dscc1', 'Dtl', 'E2f8', 'Exo1', 'Fen1',
       'Gins2', 'Gmnn', 'Hells', 'Mcm2', 'Mcm4', 'Mcm5', 'Mcm6', 'Mlf1ip',
       'Msh2', 'Nasp', 'Pcna', 'Pola1', 'Pold3', 'Prim1', 'Rad51',
       'Rad51ap1', 'Rfc2', 'Rpa2', 'Rrm1', 'Rrm2', 'Slbp', 'Tipin',
       'Tyms', 'Ubr7', 'Uhrf1', 'Ung', 'Usp1', 'Wdr76'])
G2M_genes_mouse = np.array(['Anln', 'Anp32e', 'Aurka', 'Aurkb', 'Birc5', 'Bub1', 'Cbx5',
       'Ccnb2', 'Cdc20', 'Cdc25c', 'Cdca2', 'Cdca3', 'Cdca8', 'Cdk1',
       'Cenpa', 'Cenpe', 'Cenpf', 'Ckap2', 'Ckap2l', 'Ckap5', 'Cks1b',
       'Cks2', 'Ctcf', 'Dlgap5', 'Ect2', 'Fam64a', 'G2e3', 'Gas2l3',
       'Gtse1', 'Hjurp', 'Hmgb2', 'Hmmr', 'Hn1', 'Kif11', 'Kif20b',
       'Kif23', 'Kif2c', 'Lbr', 'Mki67', 'Ncapd2', 'Ndc80', 'Nek2',
       'Nuf2', 'Nusap1', 'Psrc1', 'Rangap1', 'Smc4', 'Tacc3', 'Tmpo',
       'Top2a', 'Tpx2', 'Ttk', 'Tubb4b', 'Ube2c']) 

S_genes_human = np.array([i.upper() for i in S_genes_mouse])
G2M_genes_human = np.array([i.upper() for i in G2M_genes_mouse])

def get_cycling_gene_set(size="Medium", species="Human"):
    """
    Retrieve a marker set of cycling genes based on specified size and species.

    This function selects a predefined set of cycling genes from a collection
    categorized by size (Small, Medium, Large), each coming from a different 
    literature source. It then formats the genes according to the species 
    specified (Human, Mouse), ensuring gene names are in the appropriate case.

    Parameters:
    size (str): The size of the gene set to retrieve. Valid options are 'Small',
                'Medium', or 'Large'. Default is 'Medium'.
    species (str): The species for which the gene set is required. Valid options
                   are 'Human' or 'Mouse'. Default is 'Human'.

    Returns:
    numpy.ndarray: An array of gene names formatted according to the specified
                   species.

    Raises:
    ValueError: If an invalid size is provided (not 'Small', 'Medium', or 'Large').
    ValueError: If an invalid species is provided (not 'Human' or 'Mouse').

    Example:
    >>> get_cycling_gene_set(size="Small", species="Mouse")
    numpy.array([...]) # Returns the small set of cycling genes for mouse
    """
    if size == "Small":
        gene_set = SMALL_CYCLING_GENE_SET
    elif size == "Medium":
        gene_set = MEDIUM_CYCLING_GENE_SET
    elif size == "Large":
        gene_set = LARGE_CYCLING_GENE_SET
    else:
        raise ValueError(f"{size=} is not a valid entry. Use 'Small', 'Medium', or 'Large'.")

    if species == "Human":
        gene_set = np.array([i.upper() for i in gene_set])
    elif species != "Mouse":
        raise ValueError(f"{species=} is not a valid entry. Use 'Human' or 'Mouse'.")

    return gene_set                            
                                   
def torch_fourier_basis(ϕ, num_harmonics, der=0, device=torch.device("cpu")):
    """
    Generate a Fourier basis or its derivative based on the input angle ϕ.

    Parameters:
    ϕ (torch.Tensor): A 1D tensor representing the input angles for the Fourier basis.
    num_harmonics (int): The number of harmonics to include in the basis.
    der (int, optional): Specifies whether to return the basis (0) or its derivative (1). Default is 0.
    device (torch.device, optional): The device on which to create the tensor (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns:
    torch.Tensor: The resulting Fourier basis tensor.

    Raises:
    ValueError: If an invalid value for `der` is provided (not 0 or 1).

    Example:
    >>> torch_fourier_basis(torch.tensor([0, np.pi/4]), 3)
    # Returns the Fourier basis tensor for the specified angles and harmonics.
    """
    idx_harm = torch.concat([torch.tensor([0.0], device=device),
                             torch.repeat_interleave(torch.arange(1, 1 + num_harmonics), 2).to(device)])
    sin_cos_bool = torch.tensor([False] + [False, True] * num_harmonics, device=device)
    base_bool = torch.tensor([True] + [False] * (num_harmonics * 2), device=device)
    if der == 0:
        return torch.where(base_bool,
                             torch.tensor(1.0, dtype=torch.float32, device=device),
                             torch.where(sin_cos_bool,
                                         torch.cos(idx_harm * ϕ.unsqueeze(-1)),
                                         torch.sin(idx_harm * ϕ.unsqueeze(-1))))
    elif der == 1:
        return torch.where(base_bool,
                        torch.tensor(0.0, dtype=torch.float32, device=device),
                        torch.where(sin_cos_bool,
                                    -idx_harm * torch.sin(idx_harm * ϕ.unsqueeze(-1)),
                                     idx_harm * torch.cos(idx_harm * ϕ.unsqueeze(-1))))
    else:
        raise ValueError(f"Value {der=} is not allowed, use 0 or 1 instead")

def torch_basis(x, der=0, kind="fourier", device=torch.device("cpu"),  **kwargs, ):
    """
    A function to provide access to different kind of basis functions. 
    At the moment, only "fourier" is implemented as a valid option.

    This function supports calculating either the basis itself or its derivative.

    Parameters:
    x (torch.Tensor): A 1D tensor representing the input values for the basis functions.
    der (int, optional): Whether to compute the derivative (1) of the basis or the basis itself (0). Default is 0.
    kind (str, optional): The type of basis to use. Default is 'fourier'.
    device (torch.device, optional): The device on which to create the tensor. Default is 'cpu'.
    **kwargs: Additional keyword arguments required for specific types of bases.

    Returns:
    torch.Tensor: The resulting basis function tensor.

    Example:
    >>> torch_basis(torch.tensor([0, 1]), kind="fourier", num_harmonics=1)
    # Returns the Fourier basis tensor for the specified input and harmonics.
    """
    
    if kind == "fourier":
        if "num_harmonics" not in kwargs:
            raise ValueError("num_harmonics needs to be provided if kind=`fourier`")
        return torch_fourier_basis(x, num_harmonics=kwargs["num_harmonics"], der=der, device=device).to(device)
    else:
        raise ValueError(f"{kind=} is not a valid entry use `fourier`")

def unpack_direction(loc, concentration=1.0):
    """
    Convert a location parameter to a 2D directional vector. 
    
    This function takes an angle (or tensor of angles) and converts them to corresponding
    2D directional vectors on a unit circle, scaled by the specified concentration.

    Parameters:
    loc (torch.Tensor): A tensor of angles (in radians).
    concentration (float, optional): A scaling factor for the directional vectors. Default is 1.0.

    Returns:
    torch.Tensor: A tensor of 2D directional vectors corresponding to the input angles.

    Example:
    >>> unpack_direction(torch.tensor([0, np.pi/2]))
    # Returns the 2D directional vectors for the specified angles.
    """
    return torch.stack([torch.cos(loc), torch.sin(loc)], dim=-1) * concentration

def pack_direction(xy_pair):
    """
    Convert a 2D directional vector to an angle.

    This function takes a 2D vector (or a tensor of 2D vectors) and converts them to 
    the corresponding angles in radians. This is the inverse operation of `unpack_direction`.

    Parameters:
    xy_pair (torch.Tensor): A tensor of 2D vectors.

    Returns:
    torch.Tensor: A tensor of angles (in radians) corresponding to the input vectors.

    Example:
    >>> pack_direction(torch.tensor([[1, 0], [0, 1]]))
    # Returns the angles in radians for the specified 2D vectors.
    """
    _cos, _sin = xy_pair[..., 0], xy_pair[..., 1]
    return torch.atan2(_sin, _cos)

def simulate_data(Nc=5000, Ng=500, omegas_to_test=[0.4]):
    mv_means = np.array([0.4, 0.00, 0.0, 0.0, 2.0])
    corr_matrix = np.array([[1.0, 0.05, 0.05, 0.05, 0.30], 
                       [0.05, 1.0, 0.0, 0.0, 0.0], 
                       [0.05, 0.0, 1.0, 0.0, 0.0], 
                       [0.05, 0.0, 0.0, 1.0, 0.30], 
                       [0.30, 0.0, 0.0, 0.30, 1.0]])

    std_devs = np.array([1.2, 0.2, 0.2, 0.5, 1.0])
    mv_cov_matrix = np.diag(std_devs) @ corr_matrix @ np.diag(std_devs)
    
    simulated_phis = torch.stack([torch.tensor(np.random.uniform(0, np.pi*2)) for i in range(Nc)])
    simulated_ζ = torch.stack([utils.torch_basis(simulated_phis[i], der=0, kind="fourier", **dict(num_harmonics=1)).T for i in range(Nc)])
    simulated_ζ_dϕ = utils.torch_basis(simulated_phis.squeeze(), der=1, kind="fourier", **dict(num_harmonics=1))

    simulated_velo_parameters = torch.stack([pyro.sample("nu", dist.MultivariateNormal(loc=torch.tensor(mv_means).float().T, 
                                                                                       covariance_matrix=torch.tensor(mv_cov_matrix).float())) for i in range(Ng)]).unsqueeze(-2)
    
    simulated_nu = simulated_velo_parameters[:, :, :3]
    simulated_gammas = simulated_velo_parameters[:, :, 3]
    simulated_betas = simulated_velo_parameters[:, :, 4]

    simulated_ElogS = (simulated_nu * simulated_ζ).sum(-1)

    gamma_alpha = 1.0
    gamma_beta = 2.0
    simulated_shape_inv = torch.stack([pyro.sample("shape_inv", dist.Gamma(gamma_alpha, gamma_beta)) for i in range(Ng)])

    data_dict = {}
    for curr_omega in omegas_to_test:
        print("Simulating", curr_omega)
        simulated_omega = torch.tensor(curr_omega).repeat(Nc).float()
        simulated_ElogU = -simulated_betas + torch.log(torch.relu((simulated_nu * simulated_ζ_dϕ).sum(-1) * simulated_omega + torch.exp(simulated_gammas)) + 1e-5) + simulated_ElogS

        simulatedS = torch.stack([pyro.sample("S", dist.GammaPoisson(1.0 / simulated_shape_inv, 1.0 / (simulated_shape_inv * torch.exp(simulated_ElogS[:, i])))) for i in range(Nc)]).T
        simulatedU = torch.stack([pyro.sample("U", dist.GammaPoisson(1.0 / simulated_shape_inv, 1.0 / (simulated_shape_inv * torch.exp(simulated_ElogU[:, i])))) for i in range(Nc)]).T

        data_dict[curr_omega] = {"simulatedS":simulatedS,
                                 "simulatedU":simulatedU,
                                 "simulated_ElogU":simulated_ElogU,
                                 "simulated_omega":simulated_omega}

    completeS = torch.hstack([data_dict[i]["simulatedS"] for i in omegas_to_test])
    completeU = torch.hstack([data_dict[i]["simulatedU"] for i in omegas_to_test])

    sim_data = sc.AnnData(completeS.numpy()).T
    sim_data.layers["spliced"] = csr_matrix(completeS.numpy().T)
    sim_data.layers["unspliced"] = csr_matrix(completeU.numpy().T)

    batch = np.concatenate([np.repeat(str(curr_omega), Nc) for curr_omega in omegas_to_test])
    sim_data.obs["batch"] = batch

    completeElogU = torch.hstack([data_dict[i]["simulated_ElogU"] for i in omegas_to_test])
    sim_data.layers["simulated_ElogU"] = csr_matrix(completeElogU.numpy().T)

    complete_simulated_omega = torch.hstack([data_dict[i]["simulated_omega"] for i in omegas_to_test])
    sim_data.obs["simulated_omega"] = complete_simulated_omega

    simulated_ElogS_complete = torch.hstack([simulated_ElogS for i in range(0, len(omegas_to_test))])
    sim_data.layers["simulated_ElogS"] = csr_matrix(simulated_ElogS_complete.numpy().T)

    sim_data.var["simulated_shape_inv"] = simulated_shape_inv.numpy()
    sim_data.obs["simulated_phis"] = np.hstack([simulated_phis.numpy() for i in range(0, len(omegas_to_test))])
    sim_data.var["simulated_gammas"] = simulated_gammas.numpy()
    sim_data.var["simulated_betas"] = simulated_betas.numpy()
    sim_data.uns["simulated_ζ"] = np.vstack([simulated_ζ.numpy() for i in range(0, len(omegas_to_test))])
    sim_data.uns["simulated_ζ_dφ"] = np.vstack([simulated_ζ_dφ.numpy() for i in range(0, len(omegas_to_test))])
    sim_data.uns["simulated_nu"] = np.vstack([simulated_nu.numpy() for i in range(0, len(omegas_to_test))])

    gene_names = ["G"+str(i).zfill(5) for i in range(0, Ng)]
    sim_data.var.index = gene_names

    cell_names = ["C"+str(i).zfill(5) for i in range(0, Nc)]
    cell_names = np.stack([cell_names for i in range(0, len(omegas_to_test))]).flatten()
    cell_names = ["Velo"+str(i).replace(".", "")+":"+j for i,j in zip(sim_data.obs["batch"], cell_names)]
    sim_data.obs.index = cell_names
    return sim_data

def circular_corrcoef(x1, x2):
    """
    Returns the circular correlation coefficient between two numpy arrays.

    Args:
    x1, x2 : numpy array, shape (n,)
        Two arrays containing circular data.

    Returns:
    circular_corr : float
        The circular correlation coefficient.
    """
    
    assert len(x1) == len(x2), "Input arrays must have the same length"
    
    # Convert inputs to unit circle coordinates
    x1_unit = np.exp(1j * x1)
    x2_unit = np.exp(1j * x2)

    # Compute the product of x1 and the conjugate of x2
    prod = x1_unit * np.conj(x2_unit)

    # Compute circular correlation coefficient
    circular_corr = np.abs(np.mean(prod))
    
    return circular_corr