#!/usr/bin/env python
# QC Google API function
# Written by: Alice Murphy

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

#OLD QC FOLDER QC_FOLDER='1pFdcsrBGmAvSZ9ggix-STMbwo0amcZSC'
QC_FOLDER='1Po9DiC0oMjIH1hAjyEt4_wihkLJPjOcO'
#QC_FOLDER = '1MpT5wdV1Zoujw3zrhv9KIwzc6n1jBXN4'
TEAM_DRIVE_ID='0B2OaqiAOwIqsbVdpdjJaY3ZRa0E?resourcekey=0-TvJmtwDrcLrpqwAp3l5WWw'
# [OLD] The following dictionary lists the paths to specific items / locations on google drive - the unique IDs at the end of the links are unchanged when filenames change.
# To change or add new links, just copy the link from your browser while inside the folder. Delete the "u/1/" from the path after drive, before folders
# [OLD] links:
#LINKDICTIONARY={'pointer':'https://www.googleapis.com/drive/folders/1Jjr0dU9MNm_giJ4-s_mVKSibrqKkVaqQ',
#                'pointer_fbb':'https://www.googleapis.com/drive/folders/1E5XPhtWpI0i3-Ngl5OhpZd1wJriKewDm',
#                'pointer_mk6240':'https://www.googleapis.com/drive/folders/1S6B_IaIWwVmOEzW2_ZUPZEbyBlih5QL5',
#                'pointer_fbb_needs_qc':'https://drive.google.com/drive/folders/1pRaETk6wixMoaXwjrMDPX_cz6G4T2VMi',
#                'pointer_mk6240_needs_qc':'https://drive.google.com/drive/folders/1USTyb_ARYDKj4TyNoeKeQnaHsB3Pp1DX',
#                'pointer_fbb_v1':'https://drive.google.com/drive/folders/1-OOaft0MnYPZsGVZgYbOCXJ1xsU5848b',
#                'pointer_mk6240_v1':'https://drive.google.com/drive/folders/1wF43hpIn1D6Q7g6Z2uXSi_CeohsYdCi8',
#                'adni':'https://www.googleapis.com/drive/folders/1tZXpZjYcCtZTcpU9cgv_GIS3PDMYlcsv',
#                'adni_fbp':'https://www.googleapis.com/drive/folders/1nGC6JtkLXh7aIVwcHCLvea1zQDkH_8f3',
#                'adni_fbp_v1':'https://drive.google.com/drive/folders/1pACpcWdoEU9oL4aNHsoBaYReqGtqRptX',
#                'adni_fbp_v2':'https://drive.google.com/drive/folders/1dUHcuAVRFhvQW_XIQDJ8gi0I6gx687Hd',
#                'adni_fbp_v3':'https://drive.google.com/drive/folders/1INMYG9wLuOTTdR-iBAb79x2OkzU4mqy-',
#                'adni_fbp_v4':'https://drive.google.com/drive/folders/12WKmI01qaxdlrlLzTbzIxe8U3W549L8X',
#                'adni_fbp_v5':'https://drive.google.com/drive/folders/14kWktamiMqtsUmQP1cT-vcGQHC4YDPDv',
#                'adni_fbp_v6':'https://drive.google.com/drive/folders/1fb5xOoKjlvL8fR7Vv3aJlsMOf3xHoM4O',
#                'adni_fbp_needs_qc':'https://drive.google.com/drive/folders/1AtWsS1Q8ER4-dZUz4DbRJx_WSPtKLiBH',
#                'adni_fbb':'https://www.googleapis.com/drive/folders/1JhH6b0GKdKnRp5MsqKvjtBbAyMRlSrZb',
#                'adni_fbb_v1':'https://drive.google.com/drive/folders/1htuHtWa9FJnKopEG11GzTdqnXxibY9e3',
#                'adni_fbb_v2':'https://drive.google.com/drive/folders/1qTPPJbvlkEMYtFgby06BdafNnw6fGHjS',
#                'adni_fbb_needs_qc':'https://drive.google.com/drive/folders/11c-luA-MKTSHdmDZApfOza1a_EFBjA8y',
#                'adni_ftp':'https://www.googleapis.com/drive/folders/1GhoCEXeZYt5SlItFm2ey5JNJRoYSrd_D',
#                'adni_ftp_v1':'https://drive.google.com/drive/folders/1hNozLOd3SILzKgBlOvWgvy84ileCl4s9',
#                'adni_ftp_v2':'https://drive.google.com/drive/folders/19ZNhWKv9mYkCwpTFrK5AY9SBPApfBUqT',
#                'adni_ftp_v3':'https://drive.google.com/drive/folders/1yswgC1GGTISA4XQNt2u813Z5DbdhUikt',
#                'adni_ftp_v4':'https://drive.google.com/drive/folders/1aSndIGoryB-Gl_UTqd1BTMYYdSWhv87s',
#                'adni_ftp_v5':'https://drive.google.com/drive/folders/1k74TRIy9x5rh6wXgcTcUintzhEQ1h1eu',
#                'adni_ftp_needs_qc':'https://drive.google.com/drive/folders/1BqJ0K0IR_2e-YpIVwaxx2AvBO1j3WAmZ'}
# [OLD] links:
#LINKDICTIONARY={\
#                'adni_fbp-suvr'   :'https://drive.google.com/drive/folders/1g_i8NQVMSzcaI-wP68_i9bAZ-IXVlEvU',
#                'adni_fbp-suvr_v1':'https://drive.google.com/drive/folders/1vbxiaWsIMjIiLE9bo1Xm8utEUj0FkAYM',
#                'adni_fbp-suvr_v2':'https://drive.google.com/drive/folders/1Pnw8JNfNiQa0t89mYeIOSlXGnr64myPm',
#                'adni_fbp-suvr_v3':'https://drive.google.com/drive/folders/1nql9FefV2-u9ewz8TL4n0EyeCFUdZwve',
#                'adni_fbp-suvr_v4':'https://drive.google.com/drive/folders/1tAn5L3bzWyHVDZGZVrfmaXH8rCJaIB4t',
#                'adni_fbp-suvr_v5':'https://drive.google.com/drive/folders/1WljpsjLu7zplsi6fXTnZwetiULk8NFZI',
#                'adni_fbp-suvr_v6':'https://drive.google.com/drive/folders/1zeG7SrYpHi-lIqCWgyLDQyASXHmhbVgP',
#                'adni_fbp-suvr_v7':'https://drive.google.com/drive/folders/1SVxC-cM72CC6hnocTqV-GImUHoIXPu9g',
#                'adni_fbp-suvr_v8':'https://drive.google.com/drive/folders/1sOFrX1WZwQHwBu7q8dZdD-jwOO7mqsgm',
#                'adni_fbp-suvr_needs_qc':'https://drive.google.com/drive/folders/1EseqbCMUHzmWHbdOTZolbd7UHaivKImX',
#                'adni_ftp-suvr'   :'https://drive.google.com/drive/folders/1-HLXpni2SDbeSK6q5xjusFqHlMSaRJtd',
#                'adni_ftp-suvr_v1':'https://drive.google.com/drive/folders/1t2RxwY78MGZ-YjDlhyuBRDSgTjgHnul8',
#                'adni_ftp-suvr_v2':'https://drive.google.com/drive/folders/1c9kcrmwwfrfhRC8Ni7pE7g_F6L6hi_SG',
#                'adni_ftp-suvr_v3':'https://drive.google.com/drive/folders/1Iu8nVQmIYoyVwHil_Ec640VweOm9Nwol',
#                'adni_ftp-suvr_v4':'https://drive.google.com/drive/folders/1HjWCQvUIvEJu7OnFmj3DW3SxtOTJ-nJU',
#                'adni_ftp-suvr_v5':'https://drive.google.com/drive/folders/1q037m91WKOoNYeltEuMY6YfXiA06FGep',
#                'adni_ftp-suvr_v6':'https://drive.google.com/drive/folders/15n7uxK2gWrHOEYCC94_7pVWAg_PmwA7r',
#                'adni_ftp-suvr_v7':'https://drive.google.com/drive/folders/19HQrafvLC8C4oahPRFsfKLod1AnU8Bp-',
#                'adni_ftp-suvr_v8':'https://drive.google.com/drive/folders/1FQ-nrFe-tTyvmxhQnl9XPDGAxtJtOdo5',
#                'adni_ftp-suvr_needs_qc':'https://drive.google.com/drive/folders/1oQKPSOyPqDjtg3t_UNA_CU2jyqmcYgVl',
#                'adni_fbb-suvr'   :'https://drive.google.com/drive/folders/1eXzIVh_WH-X0MKTDrlVGOyoCQy12oEr6',
#                'adni_fbb-suvr_v1':'https://drive.google.com/drive/folders/1DUnX0ryUhc04QYUT_w4Ut1XytiDQOlKw',
#                'adni_fbb-suvr_v2':'https://drive.google.com/drive/folders/1dpnp-SrtHaxUnBk1Siv42tZaze5M8mbS',
#                'adni_fbb-suvr_v3':'https://drive.google.com/drive/folders/1vPG3U2gH2jg9LIvLKa0DAmAzQ9EVcMDc',
#                'adni_fbb-suvr_v4':'https://drive.google.com/drive/folders/1ajxYfySY65VKcvajLJCyOZ7-m1f9FpMN',
#                'adni_fbb-suvr_v5':'https://drive.google.com/drive/folders/1ZMNPdElBIkgKcLGYE_JPs618Nz2dyjdy',
#                'adni_fbb-suvr_v6':'https://drive.google.com/drive/folders/1Nnjl49cIxFlr0xBPGx5q5J3b4WWf642P',
#                'adni_fbb-suvr_v7':'https://drive.google.com/drive/folders/1pKvOLkeWHQzPZ3PwqneRMbyhJemZ5nzW',
#                'adni_fbb-suvr_v8':'https://drive.google.com/drive/folders/1885rPeTw94fL716A2sOVMVzl1r9ERQ1K',
#                'adni_fbb-suvr_needs_qc':'https://drive.google.com/drive/folders/1yejr8X6aFsnSX8WNMeAVGoJ63OyRSSTg',
#                'bacs_ftp-suvr'   :'https://drive.google.com/drive/folders/1-evwgZKIEQFotJhf_TOIsccdIREQdZl5',
#                'bacs_ftp-suvr_v1':'https://drive.google.com/drive/folders/1OYsBZiUYt5nXs2OsMrivEdUctYBDti0g',
#                'bacs_ftp-suvr_v2':'https://drive.google.com/drive/folders/119yXcMD_yM2hEFYXsX_eFZNEhiCBpOnF',
#                'bacs_ftp-suvr_v3':'https://drive.google.com/drive/folders/13d_JLyV2ECAfmv2mHZVwXKuwel_08tWI',
#                'bacs_ftp-suvr_v4':'https://drive.google.com/drive/folders/1yj39uvsr5RSH-E2DiN5T7puWcNX0YLyk',
#                'bacs_ftp-suvr_v5':'https://drive.google.com/drive/folders/1v9UQbC1w5-kndXcZRWWMXM8avqn3IH_9',
#                'bacs_ftp-suvr_v6':'https://drive.google.com/drive/folders/1R_TFGS0YBWx4NdhdHgiz0kyMo2ZCCfjx',
#                'bacs_ftp-suvr_v7':'https://drive.google.com/drive/folders/119yXcMD_yM2hEFYXsX_eFZNEhiCBpOnF',
#                'bacs_ftp-suvr_v8':'https://drive.google.com/drive/folders/1kTVuOoXHJRWPIS5x8vqI2iJ0ug5_ofSs',
#                'bacs_ftp-suvr_needs_qc':'https://drive.google.com/drive/folders/1oM-RKvF8DOxvFfACaNBqinEW9ChYWZAE',
#                'bacs_pib-dvr'   :'https://drive.google.com/drive/folders/1LTJ7oAhO1KAcUcx0odObXmE6paQskJhT',
#                'bacs_pib-dvr_v1':'https://drive.google.com/drive/folders/1-23u5VOEY0jdMIBqt2hSZeqfWruoMyAN',
#                'bacs_pib-dvr_v2':'https://drive.google.com/drive/folders/13mRKaNvv5y6H4PsTvc4CQzurdfaKKwlo',
#                'bacs_pib-dvr_v3':'https://drive.google.com/drive/folders/1uQuYtze4u6FJW04EdA38QT8MFMK2vr-j',
#                'bacs_pib-dvr_v4':'https://drive.google.com/drive/folders/1ANMJ-4Vn9OFDa9cBKDbcC6yagiMaRUHO',
#                'bacs_pib-dvr_v5':'https://drive.google.com/drive/folders/1O0lS0EV805uck1TNxMVm-mJ0VRk8NTb1',
#                'bacs_pib-dvr_v6':'https://drive.google.com/drive/folders/1uQZUyrlDU0-bDcScNsw1JkdcmgOrVRz8',
#                'bacs_pib-dvr_v7':'https://drive.google.com/drive/folders/1bRqgm90cefK4kY_JI8q_h-7eDS9vRt37',
#                'bacs_pib-dvr_v8':'https://drive.google.com/drive/folders/1g4Q9eEHKwG2wvKhYCk4SE_3hOEF1AS3M',
#                'bacs_pib-dvr_needs_qc':'https://drive.google.com/drive/folders/1Aff-NGkpWqrqmNmG2mgMKOZJJ3PKsqqu',
#                'bacs_pib-suvr'   :'https://drive.google.com/drive/folders/1-hGXf6zgkO586BXTQnSPBWCSlgXyXPS7',
#                'bacs_pib-suvr_v1':'https://drive.google.com/drive/folders/1Fn-xDCe3Y5YJ2t5wU6-WLUux4hY3F3vJ',
#                'bacs_pib-suvr_v2':'https://drive.google.com/drive/folders/1ZHHjMsVZeinGUYlBHLtAD6Zqsh1RRBwz',
#                'bacs_pib-suvr_v3':'https://drive.google.com/drive/folders/1t6kuhfThenWI4FCdJmGhNxUMFYu38hyi',
#                'bacs_pib-suvr_v4':'https://drive.google.com/drive/folders/18WBbMi_m4SO_vvr_aDC7q7tJUmRSc94N',
#                'bacs_pib-suvr_v5':'https://drive.google.com/drive/folders/1k01s0Gi4OcjxFPbj1ZE6MdnFY5_bBFWP',
#                'bacs_pib-suvr_v6':'https://drive.google.com/drive/folders/10O1JNxhMyiW2faS8CXZazSfhsNOVjBDm',
#                'bacs_pib-suvr_v7':'https://drive.google.com/drive/folders/1nNT3je7YAiROOA94thQZiLd5OFIps0S5',
#                'bacs_pib-suvr_v8':'https://drive.google.com/drive/folders/1AO9r0lD3bTdrmQD-JtcMjBY3iYtC8ryZ',
#                'ucsf_ftp-suvr'   :'https://drive.google.com/drive/folders/1GORUsLM0ZKBndl6chFd6WdHMHxgn5Ztg',
#                'ucsf_ftp-suvr_v1':'https://drive.google.com/drive/folders/1mjJH7Y2PJVLVXN3-LmtqAC3LNp2E1FqE',
#                'ucsf_ftp-suvr_v2':'https://drive.google.com/drive/folders/1hf_sNCZ7edf3e3WIaKezX5G-fAdLAyzj',
#                'ucsf_ftp-suvr_v3':'https://drive.google.com/drive/folders/1Bt7UMwodj79k_D6T-PeoMjRzFIgnPYXX',
#                'ucsf_ftp-suvr_v4':'https://drive.google.com/drive/folders/1ti3TfAKjAUIqS4YK5bRiCxjCKo6x9l_0',
#                'ucsf_ftp-suvr_v5':'https://drive.google.com/drive/folders/1nZvWdLeaddsuMVDKxgMYidun_93XhQm7',
#                'ucsf_ftp-suvr_v6':'https://drive.google.com/drive/folders/1aHdXMM5WO6Y4TVduMYdqBPaWh__pzX7K',
#                'ucsf_ftp-suvr_v7':'https://drive.google.com/drive/folders/1hkspj7-TBGqSRUov0dPrOk8u-58DTdLS',
#                'ucsf_ftp-suvr_v8':'https://drive.google.com/drive/folders/1jLHHgzFGAAwZ9B3e4S072iHisGI4-iHh',
#                'ucsf_ftp-suvr_needs_qc':'https://drive.google.com/drive/folders/1ViWX2-cRBW-LHbAJ64LQNqBbrW5J_GFR',
#                'ucsf_fbp-suvr'   :'https://drive.google.com/drive/folders/1-ThgV5_Uj5yxZxC0MoXOkcMNlsDGAyKI',
#                'ucsf_fbp-suvr_v1':'https://drive.google.com/drive/folders/1cY_Sty9Sq4AjUMyrRSADvgx1aU7p7Mdr',
#                'ucsf_fbp-suvr_v2':'https://drive.google.com/drive/folders/1ubpJtfUYJKYIL7PhuwhCJAfeiNuQroBd',
#                'ucsf_fbp-suvr_v3':'https://drive.google.com/drive/folders/1kHsSY_EHmaD0hZh2eQiKJGOob4ZjqEN7',
#                'ucsf_fbp-suvr_v4':'https://drive.google.com/drive/folders/1aMm0Cz4VFVULTYaFDJXQ1uZVAnge4X4a',
#                'ucsf_fbp-suvr_v5':'https://drive.google.com/drive/folders/1NB-rY4lBZe7sSPkA8NT_jIQBBWspjiYz',
#                'ucsf_fbp-suvr_v6':'https://drive.google.com/drive/folders/1AiDL1m7W_IK4XiHyD47s_DefJ8xnMpMv',
#                'ucsf_fbp-suvr_v7':'https://drive.google.com/drive/folders/1KoITx2d9LoNfXRCd9KhdImshCvpzx_tO',
#                'ucsf_fbp-suvr_v8':'https://drive.google.com/drive/folders/1PcxDTrLtO_bdLLUB0tAVEz88lDeAb_Xh',
#                'ucsf_fbp-suvr_needs_qc':'https://drive.google.com/drive/folders/1prOJOCAM7Wm3jsK7havkepo7klOHmkeU',
#                'ucsf_pib-suvr'   :'https://drive.google.com/drive/folders/1-YwJHsEwq4oxn3qxIMTxh16vCxmUKKxL',
#                'ucsf_pib-suvr_v1':'https://drive.google.com/drive/folders/1pOgU0smVYKG90rYd1x_aAK70BwI0Lg6U',
#                'ucsf_pib-suvr_v2':'https://drive.google.com/drive/folders/1agwCI6ejxfca6CxI9EKncE7Yt5Tzl_lA',
#                'ucsf_pib-suvr_v3':'https://drive.google.com/drive/folders/1UYc_MkSyLouvF89IbzmzTXCY3SmGd0vu',
#                'ucsf_pib-suvr_v4':'https://drive.google.com/drive/folders/1NyNY7STf0-LCdcx81t-MBWba-eZ6Z3ja',
#                'ucsf_pib-suvr_v5':'https://drive.google.com/drive/folders/16sdNlO20XgeJe_wHn63Cc1TPHs1ZxJz7',
#                'ucsf_pib-suvr_v6':'https://drive.google.com/drive/folders/1VOeXu9NRAHt6LpDbO_l7qNXsbz4tMU61',
#                'ucsf_pib-suvr_v7':'https://drive.google.com/drive/folders/1ChIsq9aMc1XPLnC0YJ8zSM0lXQpWcNNj',
#                'ucsf_pib-suvr_v8':'https://drive.google.com/drive/folders/17eeMiJTLPQO8xQejn4X0zZgtJxnVIVFT',
#                'ucsf_pib-dvr_needs_qc':'https://drive.google.com/drive/folders/11h-zt_neRbGY-W2ws-XY-siUSb-TvbWI',
#                'ucsf_pib-dvr'   :'https://drive.google.com/drive/folders/1Dora0u58_EmHf-iBq7ss3ZN_NGH-cDre',
#                'ucsf_pib-dvr_v1':'https://drive.google.com/drive/folders/15XxyFEX82cBHBDvWdbGyq9YmIie9Ks-f',
#                'ucsf_pib-dvr_v2':'https://drive.google.com/drive/folders/1bEL4cpj79B_X4UE_jiz-jELAixDLahBA',
#                'ucsf_pib-dvr_v3':'https://drive.google.com/drive/folders/16kjtq5WTFtX8JyQ-A-URCYNkMd5ipo6g',
#                'ucsf_pib-dvr_v4':'https://drive.google.com/drive/folders/1znnAfQjQSjf-JFjpKfyRkoWBc_S66_o4',
#                'ucsf_pib-dvr_v5':'https://drive.google.com/drive/folders/1dbwLaf15AQwGJHHrtuQ_eezg0C4bOFrb',
#                'ucsf_pib-dvr_v6':'https://drive.google.com/drive/folders/17dJDViLeG3rQYIZOfK5mwYSlkpNz8eS5',
#                'ucsf_pib-dvr_v7':'https://drive.google.com/drive/folders/1nlaiRzH1cip3USE-KX6VCtmmRklbG8K_',
#                'ucsf_pib-dvr_v8':'https://drive.google.com/drive/folders/1sxS6TyqZ1RXU7zhuFf0vuqcPwh7H8XHx',
#                'ucsf_pib-dvr_needs_qc':'https://drive.google.com/drive/folders/14COO-6YwTVpZWGBo4o0yrHSl6LrMXzt1',
#                'pointer_mk6240-suvr'   :'https://drive.google.com/drive/folders/12bujrVGfNkB4Xq9wgfEyqH6DCgrzHzAM',
#                'pointer_mk6240-suvr_v1':'https://drive.google.com/drive/folders/1GbVzwClwqILJ9GM7xW4IXT34rYmRvjqb',
#                'pointer_mk6240-suvr_v2':'https://drive.google.com/drive/folders/1nvbveo542wmm5-HkMBk5QBNMWHnJo9Va',
#                'pointer_mk6240-suvr_v3':'https://drive.google.com/drive/folders/1nk6DynQ3PPqa-6ivMHOIt82NOVWGLVt8',
#                'pointer_mk6240-suvr_v4':'https://drive.google.com/drive/folders/1R6Ma1up0-1wW1PStkBQgvTfKGJavCU2N',
#                'pointer_mk6240-suvr_v5':'https://drive.google.com/drive/folders/1R3ItenkK9lTt7Fo_mjmdJY6WuibSPuRc',
#                'pointer_mk6240-suvr_v6':'https://drive.google.com/drive/folders/1mvpWmDgg8Su3meBLFIwpdNAjp4UURM9D',
#                'pointer_mk6240-suvr_v7':'https://drive.google.com/drive/folders/1u1q0kB4w5osdSw4G4UCN6SH0aXyJ1z3N',
#                'pointer_mk6240-suvr_v8':'https://drive.google.com/drive/folders/1ybZD0PeIokOt8c-k0qQoj_l-lO-JiKjf',
#                'pointer_mk6240-suvr_needs_qc':'https://drive.google.com/drive/folders/1pZKQI5VlwxOBmj3BZ8aX3s2UL9yeYIoy',
#                'pointer_fbb-suvr'   :'https://drive.google.com/drive/folders/1XAWrDPj3gKVds07Oljn394oMhmcofjgS',
#                'pointer_fbb-suvr_v1':'https://drive.google.com/drive/folders/1EuwvI4ZGgbM-TfXJo4zcsF4c8OG09dfP',
#                'pointer_fbb-suvr_v2':'https://drive.google.com/drive/folders/1zuhIi40zNGvEsEcLNFxfBNN_aqXEzt4f',
#                'pointer_fbb-suvr_v3':'https://drive.google.com/drive/folders/1ZXotU4ym8JqNSfyLJS-n1o2IgP_kVjaE',
#                'pointer_fbb-suvr_v4':'https://drive.google.com/drive/folders/1IJayFBFa7W_dES2BjUw6Vi306W4g4UXh',
#                'pointer_fbb-suvr_v5':'https://drive.google.com/drive/folders/1FgsiPMvqqxA9kZJ-kGIoeioR91kaQ4Qa',
#                'pointer_fbb-suvr_v6':'https://drive.google.com/drive/folders/1GtV52CZAJE-jVJoM60C3BDocT758hShy',
#                'pointer_fbb-suvr_v7':'https://drive.google.com/drive/folders/17MVXUzQ3_-tiNYtbRYkf9_KYxRd_JTS7',
#                'pointer_fbb-suvr_v8':'https://drive.google.com/drive/folders/17f0GLhXL5wdGMaZDoSeQtuiE347mvYTw',
#                'pointer_fbb-suvr_needs_qc':'https://drive.google.com/drive/folders/1N40_4XB1VFaUJ7cytUtKkj3IIZ9A4-Fa',
#                'scan-free_ftp-suvr'            :'https://drive.google.com/drive/folders/10IV6oM_tQGXOn9fjnjoufr_ifahov3jV',
#                'scan-free_ftp-suvr_v1'         :'https://drive.google.com/drive/folders/1dfHzfLs-Qrtp0e6KtgpuKeE08DUTws2m',
#                'scan-free_ftp-suvr_v2'         :'https://drive.google.com/drive/folders/1wmhmotSOL3CdM6z3NV-WhKyGuvQgY3_i',
#                'scan-free_ftp-suvr_v3'         :'https://drive.google.com/drive/folders/1RNDyv1HloWyS0C8oeOOGehn99KL7yKeZ',
#                'scan-free_ftp-suvr_v4'         :'https://drive.google.com/drive/folders/1nwef4Q6WIZNbbA3w-L7qjT02F4D8iwUy',
#                'scan-free_ftp-suvr_v5'         :'https://drive.google.com/drive/folders/1FuAgl-yRwta5melwXN_2XJ_N2dlxoTft',
#                'scan-free_ftp-suvr_v6'         :'https://drive.google.com/drive/folders/1Izekq6ZIFtCwnqb8djBRVAaNbHpEkLL9',
#                'scan-free_ftp-suvr_v7'         :'https://drive.google.com/drive/folders/1vuY1OfZMfWVGHDuEXjZEBkFrcssQqpBq',
#                'scan-free_ftp-suvr_v8'         :'https://drive.google.com/drive/folders/13e_aiR8WgRzZId0TS85TNBPZGN0o0wrR',
#                'scan-free_ftp-suvr_needs_qc'   :'https://drive.google.com/drive/folders/1x6pvan4lwij21J5Y_qPgqHZ8RshkRwTc',
#                'scan-free_mk6240-suvr'         :'https://drive.google.com/drive/folders/10CSo3CDeYjk925Gp55pWyrrLGCjsZIaw',
#                'scan-free_mk6240-suvr_v1'      :'https://drive.google.com/drive/folders/19NJ62wpne6io-UOwH98mN9-RW9ip_jli',
#                'scan-free_mk6240-suvr_v2'      :'https://drive.google.com/drive/folders/1OhK5jwOjDVTLqFBYnV-jT2jRIP2OyO67',
#                'scan-free_mk6240-suvr_v3'      :'https://drive.google.com/drive/folders/1KleqMZU02bbjtSWiu-mbJ5Cbgv0zX9lZ',
#                'scan-free_mk6240-suvr_v4'      :'https://drive.google.com/drive/folders/1mOVrlJar3KRzcTwsfx77cY8uF8-w97Ps',
#                'scan-free_mk6240-suvr_v5'      :'https://drive.google.com/drive/folders/1PQtXRXETqqa22-jIF7WW6h8Am6nP4SIy',
#                'scan-free_mk6240-suvr_v6'      :'https://drive.google.com/drive/folders/13Ev5FPzCT9DVPhUZdMtSIl5ua6TP1OP6',
#                'scan-free_mk6240-suvr_v7'      :'https://drive.google.com/drive/folders/1RmLXf07F5UdfCRFGD_jUfiTdUvJW0S_f',
#                'scan-free_mk6240-suvr_v8'      :'https://drive.google.com/drive/folders/1flupzHAxEDBHKAJehHR5VY03EIAaui4U',
#                'scan-free_mk6240-suvr_needs_qc':'https://drive.google.com/drive/folders/11VGyX6GA32ZOLWjaYs_xMGZmn9u9RbFZ',
#                'scan-free_fbb-suvr'            :'https://drive.google.com/drive/folders/10P_oCOFpR-EtaA6k1mLJ0oj3lyqHxL-y',
#                'scan-free_fbb-suvr_v1'         :'https://drive.google.com/drive/folders/1U2jr3a_1AdWQHqtKSQtiCYSTfnzBcyyy',
#                'scan-free_fbb-suvr_v2'         :'https://drive.google.com/drive/folders/11suRyvav0znGQ5Eey7zJXLt2uRurodJm',
#                'scan-free_fbb-suvr_v3'         :'https://drive.google.com/drive/folders/1VOyZlJC06sWqstXGKrE1gnefwywyeFQ1',
#                'scan-free_fbb-suvr_v4'         :'https://drive.google.com/drive/folders/1qG1FwVeNe3KvNz3aeb1V7SCZk76f8g8F',
#                'scan-free_fbb-suvr_v5'         :'https://drive.google.com/drive/folders/1AvLvs-Ns36dPcNM87u7i2Tg66QpbCJWm',
#                'scan-free_fbb-suvr_v6'         :'https://drive.google.com/drive/folders/1ZK5IGeiwxvF4W9eTLDpoqXm8iKeqRFpT',
#                'scan-free_fbb-suvr_v7'         :'https://drive.google.com/drive/folders/1jgC7OngIqPomv-mlnzsPsyNXHeh7KZ2O',
#                'scan-free_fbb-suvr_v8'         :'https://drive.google.com/drive/folders/12qHtj743J2KAE-TPTw-6BZwT8-ZYIAv-',
#                'scan-free_fbb-suvr_needs_qc'   :'https://drive.google.com/drive/folders/15bdt32MF6PKD-6jG7O-_tLrMpOXm3OP_',
#                'scan-free_pib-suvr'            :'https://drive.google.com/drive/folders/10NZBlCrNd3QPYiZ8yJNbh2NhOHd1cv_Z',
#                'scan-free_pib-suvr_v1'         :'https://drive.google.com/drive/folders/1NmCihxo3i91r8K3g4XSkFpZaKTnUje4X',
#                'scan-free_pib-suvr_v2'         :'https://drive.google.com/drive/folders/1PI-eeU0HtOBgf6B9dK7rxY3731foS9uj',
#                'scan-free_pib-suvr_v3'         :'https://drive.google.com/drive/folders/1PQUB2Goq8M3Ptn3sRFfURe8b62VdwnZn',
#                'scan-free_pib-suvr_v4'         :'https://drive.google.com/drive/folders/1z_sx-V2JkphMgiizFHKa2S9TzhbISfE-',
#                'scan-free_pib-suvr_v5'         :'https://drive.google.com/drive/folders/1RVcb4W9XKWvC3FpCh3qyY0yHrK9k7e7-',
#                'scan-free_pib-suvr_v6'         :'https://drive.google.com/drive/folders/1RX7xMdEB5KwdJB57Qu7-DnEfjkiu4tgp',
#                'scan-free_pib-suvr_v7'         :'https://drive.google.com/drive/folders/1-BjADXDFmbqKJsxHUhn6XGRyoOmgRUnN',
#                'scan-free_pib-suvr_v8'         :'https://drive.google.com/drive/folders/1qJSFJOq8_XivKoq-tekXIAKMuIWwVTwE',
#                'scan-free_pib-suvr_needs_qc'   :'https://drive.google.com/drive/folders/1Foa4NXJfCcXUe1znqEnCotb3RpMOTIDr',
#                'scan-free_nav-suvr'            :'https://drive.google.com/drive/folders/10N2xKHLwSy4FSB_KQSUdo7B4uxwLmIwD',
#                'scan-free_nav-suvr_v1'         :'https://drive.google.com/drive/folders/1UMK7YFp8D3m1D376JJcklHNL7dbTG6EV',
#                'scan-free_nav-suvr_v2'         :'https://drive.google.com/drive/folders/1J8_5PpRl7CQJdYJvpwzN4HEkdAbB-e8o',
#                'scan-free_nav-suvr_v3'         :'https://drive.google.com/drive/folders/1LTaf-U7MTt1qe4Pt9enOQVmFnCQT9TQS',
#                'scan-free_nav-suvr_v4'         :'https://drive.google.com/drive/folders/1R9bg2CEWnJebdMOp_U1cgrjYVN1UFpNX',
#                'scan-free_nav-suvr_v5'         :'https://drive.google.com/drive/folders/1u1Hz5Cku5duzurFmY-TIADsn02BkZpON',
#                'scan-free_nav-suvr_v6'         :'https://drive.google.com/drive/folders/1hKwz-hFZp9N0WtWJsHUj9jIdnTsg7rBz',
#                'scan-free_nav-suvr_v7'         :'https://drive.google.com/drive/folders/1oCMWg9dgxBdq9HbolXLDyUC88KX8QuKj',
#                'scan-free_nav-suvr_v8'         :'https://drive.google.com/drive/folders/1rJKdrSt-Uo46NL3eEcvS0TPRlYYxhsaT',
#                'scan-free_nav-suvr_needs_qc'   :'https://drive.google.com/drive/folders/1BMLCVucdgt0UlMBwKr7qWFACwhVONHah',
#                'scan-free_fbp-suvr'            :'https://drive.google.com/drive/folders/10QSbNjCFhoqXX7xyBjh5KJU0OZRE5ls2',
#                'scan-free_fbp-suvr_v1'         :'https://drive.google.com/drive/folders/1jnUmM8An-nzM63PRFV-7-oxGpILX6GO6',
#                'scan-free_fbp-suvr_v2'         :'https://drive.google.com/drive/folders/1eIJIs1Nle6TsJD3zc5MGpN-ztXUX1NIy',
#                'scan-free_fbp-suvr_v3'         :'https://drive.google.com/drive/folders/1pcXjkvLDJDo54NaSfIvF4hn3f7084pUY',
#                'scan-free_fbp-suvr_v4'         :'https://drive.google.com/drive/folders/16UAKxCnYWpeqpcFfSBFaClRPB2eFMvpO',
#                'scan-free_fbp-suvr_v5'         :'https://drive.google.com/drive/folders/1qHDxpe__x0N1zOMW2V_GKc0kpvgui2yk',
#                'scan-free_fbp-suvr_v6'         :'https://drive.google.com/drive/folders/1l4X-s8i2TBHoGJ9pM4_v7_X9_KjQQLUj',
#                'scan-free_fbp-suvr_v7'         :'https://drive.google.com/drive/folders/1btG8B8XxPQuyhTwifL0EGRQoeiOkY2nZ',
#                'scan-free_fbp-suvr_v8'         :'https://drive.google.com/drive/folders/1621zJSiHHrTlTitMJkMWkrTHQ9AgjBjm',
#                'scan-free_fbp-suvr_needs_qc'   :'https://drive.google.com/drive/folders/1iqDR10I2Ny8jK-R82TmiZ5xqlRWCwyu0',
#                }

def get_qc_folder_id(GAUTH,filepath,qc_folder_id='1MpT5wdV1Zoujw3zrhv9KIwzc6n1jBXN4'):
    '''
    filepath: bacs/ftp-suvr/v1
    '''
    folders = filepath.split('/')
    
    drive = GoogleDrive(GAUTH)
    my_v_folder_as_parent_query="'{0}' in parents and trashed = false and title = '{1}'"
    google_http = 'https://drive.google.com/drive/folders/{0}'
    current_dir_id = qc_folder_id
    for filename in folders:
        file_list = drive.ListFile({'q': my_v_folder_as_parent_query.format(current_dir_id,filename),'supportsAllDrives': True, 'includeItemsFromAllDrives': True}).GetList()
        if len(file_list) >= 2:
            print('########################################')
            print('ERROR Error error!!')
            print('')
            print(google_http.format(current_dir_id))
            print('')
            raise Exception('Delete the duplicate study folder in the google qc folder')
        
        elif len(file_list) == 0:
            print('Creating new study QC folder',filename, 'in' ,filepath)
            file_metadata = {
                          'title': filename,
                          'parents': [{"kind": "drive#parentReference", "id": current_dir_id}],
                          'supportsAllDrives': 'true',
                          'shared':'true',
                          'mimeType': 'application/vnd.google-apps.folder'
                        }
            pfolder = drive.CreateFile(file_metadata)
            pfolder.Upload(param={'supportsAllDrives': True})
            current_dir_id = pfolder['id']
        
        else:
            current_dir_id = file_list[0]['id']
            
    return current_dir_id


def initiate_auth(cred=0):
    GAUTH = GoogleAuth()
    print('Loading credentials')
    GAUTH.LoadCredentialsFile(f'petcore_credentials/petcore_creds.txt')
    CREDENTIALS=GAUTH.credentials
    if CREDENTIALS is None:
        raise ValueError('Credentials are empty!!!!!!!!!!!!!!!')
    print('Authentification successful.')
    return GAUTH,CREDENTIALS

def create_creds():
    print('do not use!!!!!!!!!')
    ####### GAUTH.SaveCredentialsFile("_mycreds.txt") ## don't uncomment, don't overwrite mycreds.txt
    return print('Process complete - do not overwrite file')
