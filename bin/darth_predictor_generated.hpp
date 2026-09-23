#pragma once

namespace quiver {

inline float predict_darth_tree(int tree_id, const float* f) {
  switch (tree_id) {
    case 0: {
      if (f[3] <= 80118.5f) {
        if (f[0] <= 17.0f) {
          if (f[3] <= 60577.5f) {
            if (f[0] <= 13.0f) {
              return 0.639237393f;
            } else {
              return 0.648274561f;
            }
          } else {
            return 0.632699128f;
          }
        } else {
          if (f[2] <= 46340.5f) {
            if (f[5] <= 4.5f) {
              return 0.65631051f;
            } else {
              return 0.66204872f;
            }
          } else {
            if (f[5] <= 3.5f) {
              return 0.647009759f;
            } else {
              return 0.65477268f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[0] <= 3.0f) {
            return 0.592284446f;
          } else {
            if (f[2] <= 54630.0f) {
              return 0.605213314f;
            } else {
              return 0.595345297f;
            }
          }
        } else {
          if (f[0] <= 7.0f) {
            if (f[2] <= 54630.0f) {
              return 0.616259736f;
            } else {
              return 0.60263604f;
            }
          } else {
            if (f[2] <= 43471.5f) {
              return 0.628067799f;
            } else {
              if (f[0] <= 19.0f) {
                return 0.617233712f;
              } else {
                return 0.636292469f;
              }
            }
          }
        }
      }
    }
    case 1: {
      if (f[3] <= 81049.0f) {
        if (f[0] <= 17.0f) {
          if (f[3] <= 57282.5f) {
            if (f[0] <= 13.0f) {
              return 0.000514137317f;
            } else {
              return 0.00899781384f;
            }
          } else {
            if (f[3] <= 69214.5f) {
              return -0.00226328529f;
            } else {
              return -0.0101374289f;
            }
          }
        } else {
          if (f[3] <= 58954.5f) {
            if (f[5] <= 4.5f) {
              return 0.0155295306f;
            } else {
              return 0.0209282504f;
            }
          } else {
            if (f[5] <= 3.5f) {
              return 0.0064378032f;
            } else {
              return 0.0137682716f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[0] <= 3.0f) {
            return -0.0431371555f;
          } else {
            return -0.0334940832f;
          }
        } else {
          if (f[0] <= 7.0f) {
            if (f[2] <= 51416.0f) {
              return -0.0206024513f;
            } else {
              return -0.0321139809f;
            }
          } else {
            if (f[2] <= 46340.5f) {
              return -0.0105869575f;
            } else {
              if (f[0] <= 17.0f) {
                return -0.0214155085f;
              } else {
                return -0.00479905528f;
              }
            }
          }
        }
      }
    }
    case 2: {
      if (f[3] <= 79325.5f) {
        if (f[0] <= 15.0f) {
          if (f[3] <= 58769.5f) {
            if (f[0] <= 11.0f) {
              return -0.00308673759f;
            } else {
              return 0.00461762574f;
            }
          } else {
            return -0.00647290287f;
          }
        } else {
          if (f[3] <= 62156.5f) {
            if (f[0] <= 21.0f) {
              return 0.011184376f;
            } else {
              if (f[5] <= 6.5f) {
                return 0.015938599f;
              } else {
                return 0.0201732452f;
              }
            }
          } else {
            if (f[0] <= 23.0f) {
              return 0.00222088534f;
            } else {
              return 0.0101250399f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[0] <= 3.0f) {
            return -0.0396861837f;
          } else {
            return -0.0308145568f;
          }
        } else {
          if (f[0] <= 7.0f) {
            if (f[2] <= 56746.5f) {
              return -0.019598839f;
            } else {
              return -0.0323343526f;
            }
          } else {
            if (f[0] <= 21.0f) {
              if (f[2] <= 54036.5f) {
                return -0.0107993939f;
              } else {
                return -0.0204947904f;
              }
            } else {
              return -0.000826957474f;
            }
          }
        }
      }
    }
    case 3: {
      if (f[3] <= 83425.0f) {
        if (f[0] <= 17.0f) {
          if (f[3] <= 65011.5f) {
            if (f[0] <= 13.0f) {
              return -0.000561149816f;
            } else {
              return 0.00663199798f;
            }
          } else {
            return -0.00741897642f;
          }
        } else {
          if (f[2] <= 49930.5f) {
            if (f[5] <= 3.5f) {
              return 0.0125277765f;
            } else {
              return 0.0172896548f;
            }
          } else {
            if (f[5] <= 2.5f) {
              return 0.0039025353f;
            } else {
              return 0.0102968269f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[0] <= 3.0f) {
            return -0.0365112886f;
          } else {
            if (f[2] <= 56504.0f) {
              return -0.0262680655f;
            } else {
              return -0.0357979514f;
            }
          }
        } else {
          if (f[2] <= 46517.0f) {
            if (f[0] <= 7.0f) {
              return -0.0166227526f;
            } else {
              return -0.00892779099f;
            }
          } else {
            if (f[3] <= 99469.0f) {
              return -0.0121629105f;
            } else {
              if (f[2] <= 59891.5f) {
                return -0.0197391057f;
              } else {
                return -0.0292934091f;
              }
            }
          }
        }
      }
    }
    case 4: {
      if (f[3] <= 79325.5f) {
        if (f[0] <= 15.0f) {
          if (f[2] <= 43471.5f) {
            if (f[0] <= 11.0f) {
              return -0.00252323901f;
            } else {
              return 0.00449064394f;
            }
          } else {
            return -0.00576394905f;
          }
        } else {
          if (f[2] <= 46340.5f) {
            if (f[0] <= 21.0f) {
              return 0.0100753146f;
            } else {
              return 0.0154734656f;
            }
          } else {
            if (f[0] <= 21.0f) {
              return 0.00184732946f;
            } else {
              return 0.00930511955f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[0] <= 3.0f) {
            return -0.0335903845f;
          } else {
            if (f[2] <= 54630.0f) {
              return -0.0239964606f;
            } else {
              return -0.0324036392f;
            }
          }
        } else {
          if (f[0] <= 7.0f) {
            if (f[2] <= 46517.0f) {
              return -0.0152929324f;
            } else {
              return -0.0240738203f;
            }
          } else {
            if (f[0] <= 23.0f) {
              if (f[2] <= 53206.5f) {
                return -0.00896391296f;
              } else {
                return -0.0169359805f;
              }
            } else {
              return 0.000665208512f;
            }
          }
        }
      }
    }
    case 5: {
      if (f[3] <= 77290.5f) {
        if (f[5] <= 1.00000002e-35f) {
          if (f[0] <= 15.0f) {
            if (f[2] <= 46340.5f) {
              return 0.00025244927f;
            } else {
              return -0.00714431194f;
            }
          } else {
            if (f[3] <= 64370.0f) {
              return 0.00728874584f;
            } else {
              return -0.000809413921f;
            }
          }
        } else {
          if (f[5] <= 3.5f) {
            if (f[3] <= 56881.5f) {
              return 0.0104581857f;
            } else {
              return 0.00546997847f;
            }
          } else {
            if (f[3] <= 56881.5f) {
              return 0.0150149525f;
            } else {
              return 0.0106618663f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[0] <= 3.0f) {
            return -0.0309031543f;
          } else {
            return -0.0239949254f;
          }
        } else {
          if (f[0] <= 7.0f) {
            if (f[2] <= 42322.5f) {
              return -0.0134847086f;
            } else {
              return -0.0213479398f;
            }
          } else {
            if (f[0] <= 19.0f) {
              if (f[2] <= 54036.5f) {
                return -0.00831471838f;
              } else {
                return -0.0159696765f;
              }
            } else {
              return -0.00112004194f;
            }
          }
        }
      }
    }
    case 6: {
      if (f[3] <= 83425.0f) {
        if (f[0] <= 19.0f) {
          if (f[3] <= 62958.5f) {
            if (f[0] <= 13.0f) {
              return -0.000356167718f;
            } else {
              if (f[5] <= 1.5f) {
                return 0.00446896782f;
              } else {
                return 0.00912632913f;
              }
            }
          } else {
            return -0.00463308207f;
          }
        } else {
          if (f[2] <= 55084.0f) {
            if (f[5] <= 6.5f) {
              return 0.0107387494f;
            } else {
              return 0.0146925003f;
            }
          } else {
            return 0.00401088047f;
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[0] <= 3.0f) {
            return -0.0284309015f;
          } else {
            if (f[2] <= 49930.5f) {
              return -0.0196956624f;
            } else {
              return -0.0268525618f;
            }
          }
        } else {
          if (f[2] <= 42107.5f) {
            if (f[0] <= 7.0f) {
              return -0.0123922567f;
            } else {
              return -0.00624491225f;
            }
          } else {
            if (f[3] <= 99469.0f) {
              return -0.00911513556f;
            } else {
              if (f[2] <= 59891.5f) {
                return -0.0148636324f;
              } else {
                return -0.0238399045f;
              }
            }
          }
        }
      }
    }
    case 7: {
      if (f[3] <= 76106.5f) {
        if (f[0] <= 19.0f) {
          if (f[5] <= 1.00000002e-35f) {
            if (f[2] <= 39097.0f) {
              if (f[0] <= 11.0f) {
                return -0.0017696479f;
              } else {
                return 0.00404514803f;
              }
            } else {
              return -0.00342951143f;
            }
          } else {
            if (f[2] <= 43471.5f) {
              return 0.00763642944f;
            } else {
              return 0.0023460393f;
            }
          }
        } else {
          if (f[2] <= 43471.5f) {
            return 0.0120228506f;
          } else {
            if (f[5] <= 5.5f) {
              return 0.00610288575f;
            } else {
              return 0.010939742f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[0] <= 3.0f) {
            return -0.0261564301f;
          } else {
            return -0.0203093046f;
          }
        } else {
          if (f[0] <= 7.0f) {
            if (f[2] <= 56746.5f) {
              return -0.0129090625f;
            } else {
              return -0.0222486704f;
            }
          } else {
            if (f[5] <= 1.00000002e-35f) {
              if (f[2] <= 49234.0f) {
                return -0.00640609716f;
              } else {
                return -0.0126708525f;
              }
            } else {
              return -0.000667012392f;
            }
          }
        }
      }
    }
    case 8: {
      if (f[3] <= 83425.0f) {
        if (f[5] <= 1.5f) {
          if (f[0] <= 13.0f) {
            if (f[2] <= 46674.0f) {
              return -0.000307944337f;
            } else {
              return -0.00801148528f;
            }
          } else {
            if (f[3] <= 65215.5f) {
              if (f[0] <= 21.0f) {
                return 0.00397840163f;
              } else {
                return 0.0083945309f;
              }
            } else {
              return -0.00136808297f;
            }
          }
        } else {
          if (f[3] <= 65011.5f) {
            if (f[5] <= 7.5f) {
              return 0.00957604795f;
            } else {
              return 0.0130571896f;
            }
          } else {
            return 0.00486883084f;
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[2] <= 56746.5f) {
            if (f[0] <= 3.0f) {
              return -0.0226801093f;
            } else {
              return -0.0170417354f;
            }
          } else {
            return -0.0259978002f;
          }
        } else {
          if (f[2] <= 42107.5f) {
            return -0.00774863787f;
          } else {
            if (f[3] <= 99469.0f) {
              return -0.00762693743f;
            } else {
              if (f[2] <= 63871.5f) {
                return -0.0131835188f;
              } else {
                return -0.0226010062f;
              }
            }
          }
        }
      }
    }
    case 9: {
      if (f[3] <= 76106.5f) {
        if (f[5] <= 1.00000002e-35f) {
          if (f[0] <= 13.0f) {
            if (f[2] <= 40290.5f) {
              return 5.42533116e-06f;
            } else {
              return -0.00562474515f;
            }
          } else {
            if (f[3] <= 55401.5f) {
              return 0.00525084994f;
            } else {
              return 0.000381410078f;
            }
          }
        } else {
          if (f[0] <= 23.0f) {
            if (f[2] <= 43471.5f) {
              return 0.00754777032f;
            } else {
              return 0.00312565682f;
            }
          } else {
            if (f[2] <= 43471.5f) {
              return 0.0113689624f;
            } else {
              return 0.00798703246f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[2] <= 59535.5f) {
            if (f[0] <= 3.0f) {
              return -0.0210055266f;
            } else {
              return -0.0158345871f;
            }
          } else {
            return -0.0245025527f;
          }
        } else {
          if (f[0] <= 7.0f) {
            return -0.0124181409f;
          } else {
            if (f[0] <= 23.0f) {
              if (f[2] <= 68329.5f) {
                return -0.00675429768f;
              } else {
                return -0.0157628498f;
              }
            } else {
              return 0.000960632174f;
            }
          }
        }
      }
    }
    case 10: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 48841.5f) {
            if (f[0] <= 3.0f) {
              return -0.0188333707f;
            } else {
              return -0.013754016f;
            }
          } else {
            return -0.0209988404f;
          }
        } else {
          if (f[2] <= 56746.5f) {
            if (f[0] <= 7.0f) {
              return -0.0101160698f;
            } else {
              if (f[2] <= 36385.5f) {
                return -0.00240598446f;
              } else {
                return -0.00670648451f;
              }
            }
          } else {
            return -0.0161155045f;
          }
        }
      } else {
        if (f[5] <= 1.5f) {
          if (f[3] <= 69214.5f) {
            if (f[0] <= 21.0f) {
              if (f[2] <= 36591.0f) {
                if (f[3] <= 8885.5f) {
                  return -0.00383472173f;
                } else {
                  return 0.00496252574f;
                }
              } else {
                return 0.00039389163f;
              }
            } else {
              return 0.00697056052f;
            }
          } else {
            return -0.00439524775f;
          }
        } else {
          if (f[3] <= 69214.5f) {
            if (f[5] <= 8.5f) {
              return 0.00805476014f;
            } else {
              return 0.0115272408f;
            }
          } else {
            return 0.00237747675f;
          }
        }
      }
    }
    case 11: {
      if (f[3] <= 76106.5f) {
        if (f[0] <= 19.0f) {
          if (f[5] <= 1.00000002e-35f) {
            if (f[2] <= 36591.0f) {
              return 0.00176214371f;
            } else {
              return -0.00252396464f;
            }
          } else {
            if (f[2] <= 40482.5f) {
              return 0.00565654099f;
            } else {
              return 0.00163549896f;
            }
          }
        } else {
          if (f[2] <= 46340.5f) {
            if (f[3] <= 7991.5f) {
              return 0.00117486691f;
            } else {
              return 0.00892861001f;
            }
          } else {
            return 0.00495901224f;
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[2] <= 60960.0f) {
            if (f[0] <= 3.0f) {
              return -0.0178295971f;
            } else {
              return -0.0134076025f;
            }
          } else {
            return -0.0211788009f;
          }
        } else {
          if (f[0] <= 7.0f) {
            if (f[2] <= 42322.5f) {
              return -0.00795522082f;
            } else {
              return -0.0137765621f;
            }
          } else {
            if (f[0] <= 21.0f) {
              if (f[2] <= 68329.5f) {
                return -0.00579835116f;
              } else {
                return -0.0146244062f;
              }
            } else {
              return 0.000242405979f;
            }
          }
        }
      }
    }
    case 12: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 45815.5f) {
            return -0.0132778534f;
          } else {
            return -0.0175928065f;
          }
        } else {
          if (f[2] <= 53206.5f) {
            if (f[0] <= 7.0f) {
              return -0.00824809587f;
            } else {
              return -0.00352986323f;
            }
          } else {
            if (f[2] <= 68329.5f) {
              return -0.0110579032f;
            } else {
              return -0.0201056746f;
            }
          }
        }
      } else {
        if (f[5] <= 1.5f) {
          if (f[3] <= 69214.5f) {
            if (f[0] <= 23.0f) {
              if (f[2] <= 36591.0f) {
                if (f[3] <= 8885.5f) {
                  return -0.00373896099f;
                } else {
                  return 0.00445003212f;
                }
              } else {
                return 0.00059547632f;
              }
            } else {
              return 0.00648199768f;
            }
          } else {
            return -0.00387343122f;
          }
        } else {
          if (f[3] <= 54993.0f) {
            if (f[3] <= 8885.5f) {
              return 0.00185078047f;
            } else {
              return 0.00843846245f;
            }
          } else {
            if (f[5] <= 6.5f) {
              return 0.00394513892f;
            } else {
              return 0.00771059017f;
            }
          }
        }
      }
    }
    case 13: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 60960.0f) {
            if (f[0] <= 3.0f) {
              return -0.0152176863f;
            } else {
              return -0.0111646608f;
            }
          } else {
            return -0.018077072f;
          }
        } else {
          if (f[2] <= 49057.5f) {
            if (f[0] <= 7.0f) {
              return -0.00725765779f;
            } else {
              return -0.00285045427f;
            }
          } else {
            if (f[2] <= 66954.0f) {
              return -0.00932502293f;
            } else {
              return -0.0176441398f;
            }
          }
        }
      } else {
        if (f[5] <= 2.5f) {
          if (f[3] <= 65215.5f) {
            if (f[0] <= 21.0f) {
              if (f[2] <= 36591.0f) {
                if (f[3] <= 8885.5f) {
                  return -0.00333109572f;
                } else {
                  return 0.00415515189f;
                }
              } else {
                return 0.000919135348f;
              }
            } else {
              return 0.00591812758f;
            }
          } else {
            if (f[0] <= 25.0f) {
              return -0.00307553481f;
            } else {
              return 0.00260938409f;
            }
          }
        } else {
          if (f[3] <= 69498.5f) {
            return 0.00743457791f;
          } else {
            return 0.00220990638f;
          }
        }
      }
    }
    case 14: {
      if (f[3] <= 74622.5f) {
        if (f[0] <= 15.0f) {
          if (f[2] <= 40856.5f) {
            return 0.000981490453f;
          } else {
            return -0.00266055408f;
          }
        } else {
          if (f[0] <= 25.0f) {
            if (f[2] <= 42666.5f) {
              if (f[3] <= 8885.5f) {
                return -0.00114388541f;
              } else {
                return 0.00566351895f;
              }
            } else {
              return 0.0018279826f;
            }
          } else {
            if (f[2] <= 42666.5f) {
              return 0.0081306763f;
            } else {
              return 0.00562276897f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[2] <= 45996.5f) {
            if (f[0] <= 3.0f) {
              return -0.0134854288f;
            } else {
              return -0.00932376558f;
            }
          } else {
            return -0.0149731356f;
          }
        } else {
          if (f[0] <= 15.0f) {
            if (f[2] <= 59270.0f) {
              if (f[1] <= 7.0f) {
                return -0.00736066991f;
              } else {
                return -0.00402295945f;
              }
            } else {
              if (f[2] <= 76519.0f) {
                return -0.0106587582f;
              } else {
                return -0.0219813486f;
              }
            }
          } else {
            return -0.00108042067f;
          }
        }
      }
    }
    case 15: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 62127.0f) {
            if (f[0] <= 3.0f) {
              return -0.0129100307f;
            } else {
              return -0.00942466039f;
            }
          } else {
            return -0.0156293334f;
          }
        } else {
          if (f[2] <= 46674.0f) {
            if (f[4] <= 1.4923265f) {
              return -0.00432751229f;
            } else {
              return 8.22093568e-06f;
            }
          } else {
            if (f[2] <= 63871.5f) {
              return -0.00754537838f;
            } else {
              return -0.0139099077f;
            }
          }
        }
      } else {
        if (f[5] <= 1.5f) {
          if (f[2] <= 55084.0f) {
            if (f[0] <= 17.0f) {
              if (f[2] <= 36744.0f) {
                return 0.00238588954f;
              } else {
                return -0.000177836351f;
              }
            } else {
              return 0.00401274214f;
            }
          } else {
            if (f[0] <= 21.0f) {
              return -0.00445941294f;
            } else {
              return 0.000208372937f;
            }
          }
        } else {
          if (f[5] <= 6.5f) {
            if (f[2] <= 55084.0f) {
              return 0.00525553236f;
            } else {
              return 0.0010108669f;
            }
          } else {
            return 0.00738263271f;
          }
        }
      }
    }
    case 16: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 41564.5f) {
            return -0.00912668057f;
          } else {
            return -0.0125120255f;
          }
        } else {
          if (f[2] <= 40482.5f) {
            if (f[0] <= 7.0f) {
              return -0.00496765702f;
            } else {
              return -0.00149578614f;
            }
          } else {
            if (f[4] <= 1.3790285f) {
              if (f[2] <= 65602.0f) {
                return -0.00751820829f;
              } else {
                return -0.0144224969f;
              }
            } else {
              return -0.00264074524f;
            }
          }
        }
      } else {
        if (f[5] <= 2.5f) {
          if (f[3] <= 69214.5f) {
            if (f[0] <= 23.0f) {
              if (f[2] <= 36591.0f) {
                if (f[3] <= 12615.5f) {
                  return -0.00184139874f;
                } else {
                  return 0.00362572348f;
                }
              } else {
                return 0.000605596555f;
              }
            } else {
              return 0.0049653946f;
            }
          } else {
            return -0.00267946842f;
          }
        } else {
          if (f[5] <= 9.5f) {
            if (f[3] <= 56306.5f) {
              return 0.00580185794f;
            } else {
              return 0.0032903454f;
            }
          } else {
            return 0.00781262408f;
          }
        }
      }
    }
    case 17: {
      if (f[3] <= 74622.5f) {
        if (f[0] <= 15.0f) {
          if (f[2] <= 47273.5f) {
            return 0.00037956335f;
          } else {
            return -0.0030622971f;
          }
        } else {
          if (f[0] <= 25.0f) {
            if (f[2] <= 46166.5f) {
              if (f[3] <= 7991.5f) {
                return -0.00205457931f;
              } else {
                return 0.00425858552f;
              }
            } else {
              return 0.000924377935f;
            }
          } else {
            if (f[2] <= 51416.0f) {
              return 0.00616399494f;
            } else {
              return 0.00358887681f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[2] <= 63871.5f) {
            if (f[0] <= 3.0f) {
              return -0.0110682424f;
            } else {
              return -0.00786732321f;
            }
          } else {
            return -0.0136707537f;
          }
        } else {
          if (f[2] <= 79786.5f) {
            if (f[0] <= 15.0f) {
              if (f[2] <= 35798.5f) {
                return -0.00276535694f;
              } else {
                if (f[4] <= 1.44255745f) {
                  return -0.00695547748f;
                } else {
                  return -0.00176060682f;
                }
              }
            } else {
              return -0.000618357637f;
            }
          } else {
            return -0.0184895266f;
          }
        }
      }
    }
    case 18: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 7.0f) {
          if (f[2] <= 40482.5f) {
            if (f[0] <= 3.0f) {
              return -0.00954411906f;
            } else {
              return -0.00513161561f;
            }
          } else {
            if (f[2] <= 65602.0f) {
              return -0.00908231228f;
            } else {
              return -0.0127954051f;
            }
          }
        } else {
          if (f[2] <= 58406.5f) {
            if (f[4] <= 1.6455195f) {
              return -0.00299651368f;
            } else {
              return 0.000992468326f;
            }
          } else {
            return -0.00889096438f;
          }
        }
      } else {
        if (f[5] <= 1.5f) {
          if (f[2] <= 54249.5f) {
            if (f[0] <= 25.0f) {
              return 0.00154354193f;
            } else {
              return 0.00486337026f;
            }
          } else {
            if (f[2] <= 65602.0f) {
              return -0.00123035126f;
            } else {
              return -0.00589332413f;
            }
          }
        } else {
          if (f[5] <= 8.5f) {
            if (f[2] <= 55084.0f) {
              if (f[3] <= 8885.5f) {
                return -0.000814250492f;
              } else {
                return 0.00447812383f;
              }
            } else {
              return 0.000982729743f;
            }
          } else {
            return 0.00643966601f;
          }
        }
      }
    }
    case 19: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 41236.5f) {
            return -0.00706046906f;
          } else {
            return -0.00982882486f;
          }
        } else {
          if (f[2] <= 56746.5f) {
            if (f[4] <= 1.4923265f) {
              if (f[2] <= 33966.5f) {
                return -0.00221555748f;
              } else {
                return -0.00480826856f;
              }
            } else {
              return 2.97330756e-05f;
            }
          } else {
            return -0.008406383f;
          }
        }
      } else {
        if (f[5] <= 1.00000002e-35f) {
          if (f[2] <= 54249.5f) {
            if (f[0] <= 25.0f) {
              return 0.000898774357f;
            } else {
              return 0.00427941632f;
            }
          } else {
            return -0.00302554522f;
          }
        } else {
          if (f[5] <= 5.5f) {
            if (f[2] <= 58406.5f) {
              if (f[3] <= 10374.5f) {
                return -0.00104983588f;
              } else {
                if (f[2] <= 35986.5f) {
                  return 0.00456469598f;
                } else {
                  if (f[0] <= 25.0f) {
                    return 0.00182995178f;
                  } else {
                    return 0.0041661822f;
                  }
                }
              }
            } else {
              return -0.000534525055f;
            }
          } else {
            return 0.00520942004f;
          }
        }
      }
    }
    case 20: {
      if (f[3] <= 72568.5f) {
        if (f[0] <= 19.0f) {
          if (f[0] <= 13.0f) {
            return -0.000941816112f;
          } else {
            if (f[2] <= 39272.0f) {
              return 0.0025399271f;
            } else {
              return 0.000270083016f;
            }
          }
        } else {
          if (f[2] <= 36591.0f) {
            if (f[3] <= 11209.5f) {
              return 0.000311405131f;
            } else {
              return 0.00524295072f;
            }
          } else {
            if (f[0] <= 27.0f) {
              return 0.0021804189f;
            } else {
              return 0.00407663504f;
            }
          }
        }
      } else {
        if (f[0] <= 7.0f) {
          if (f[0] <= 3.0f) {
            return -0.00924332066f;
          } else {
            if (f[2] <= 37400.5f) {
              return -0.00410313024f;
            } else {
              return -0.00736000926f;
            }
          }
        } else {
          if (f[2] <= 79786.5f) {
            if (f[0] <= 25.0f) {
              if (f[2] <= 64275.5f) {
                return -0.00231019183f;
              } else {
                if (f[4] <= 1.12458497f) {
                  return -0.0104428065f;
                } else {
                  return -0.00379502094f;
                }
              }
            } else {
              return 0.00180508142f;
            }
          } else {
            return -0.0158402812f;
          }
        }
      }
    }
    case 21: {
      if (f[3] <= 72568.5f) {
        if (f[0] <= 17.0f) {
          if (f[2] <= 39097.0f) {
            if (f[3] <= 9632.0f) {
              return -0.00463456798f;
            } else {
              return 0.00146685059f;
            }
          } else {
            return -0.00107837296f;
          }
        } else {
          if (f[2] <= 36591.0f) {
            if (f[3] <= 7991.5f) {
              return -0.00123447106f;
            } else {
              return 0.00454589649f;
            }
          } else {
            if (f[0] <= 25.0f) {
              return 0.00150801817f;
            } else {
              return 0.00348739489f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[2] <= 63871.5f) {
            if (f[0] <= 3.0f) {
              return -0.00802545847f;
            } else {
              return -0.00556200147f;
            }
          } else {
            return -0.0101171621f;
          }
        } else {
          if (f[0] <= 25.0f) {
            if (f[2] <= 68329.5f) {
              if (f[0] <= 7.0f) {
                return -0.00448807576f;
              } else {
                return -0.00232191425f;
              }
            } else {
              if (f[4] <= 1.13121051f) {
                return -0.011682715f;
              } else {
                return -0.00508737332f;
              }
            }
          } else {
            return 0.00154116863f;
          }
        }
      }
    }
    case 22: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 56504.0f) {
            return -0.00596410456f;
          } else {
            return -0.00861716742f;
          }
        } else {
          if (f[2] <= 46674.0f) {
            if (f[4] <= 1.6455195f) {
              return -0.00224263872f;
            } else {
              return 0.000989325841f;
            }
          } else {
            if (f[2] <= 72390.5f) {
              if (f[4] <= 1.44255745f) {
                return -0.00548565356f;
              } else {
                return -0.000652092802f;
              }
            } else {
              return -0.0123728312f;
            }
          }
        }
      } else {
        if (f[5] <= 1.00000002e-35f) {
          if (f[2] <= 51416.0f) {
            return 0.00111173674f;
          } else {
            return -0.00219525631f;
          }
        } else {
          if (f[5] <= 5.5f) {
            if (f[2] <= 58406.5f) {
              if (f[3] <= 5226.0f) {
                return -0.00385109791f;
              } else {
                if (f[0] <= 23.0f) {
                  if (f[4] <= 1.35927302f) {
                    return 0.00161223583f;
                  } else {
                    return 0.00344881847f;
                  }
                } else {
                  return 0.00363437808f;
                }
              }
            } else {
              return -0.000465431598f;
            }
          } else {
            return 0.00418061563f;
          }
        }
      }
    }
    case 23: {
      if (f[0] <= 13.0f) {
        if (f[0] <= 7.0f) {
          if (f[2] <= 51416.0f) {
            if (f[0] <= 3.0f) {
              return -0.00659340666f;
            } else {
              return -0.00378266469f;
            }
          } else {
            return -0.00733722949f;
          }
        } else {
          if (f[2] <= 49930.5f) {
            if (f[2] <= 32385.5f) {
              if (f[2] <= 6678.5f) {
                return -0.00462278202f;
              } else {
                return 0.000789523373f;
              }
            } else {
              return -0.00167934422f;
            }
          } else {
            if (f[2] <= 70062.5f) {
              return -0.00396056319f;
            } else {
              return -0.0106583572f;
            }
          }
        }
      } else {
        if (f[3] <= 56881.5f) {
          if (f[3] <= 5226.0f) {
            return -0.00379429002f;
          } else {
            if (f[0] <= 23.0f) {
              if (f[4] <= 1.40385002f) {
                return 0.00177549032f;
              } else {
                return 0.00386650566f;
              }
            } else {
              return 0.0040104847f;
            }
          }
        } else {
          if (f[0] <= 25.0f) {
            if (f[2] <= 59270.0f) {
              return 0.000599767332f;
            } else {
              return -0.00222359438f;
            }
          } else {
            return 0.00236989107f;
          }
        }
      }
    }
    case 24: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 65602.0f) {
            return -0.00526061775f;
          } else {
            return -0.00823551051f;
          }
        } else {
          if (f[2] <= 58406.5f) {
            if (f[2] <= 31129.5f) {
              return -0.000656243362f;
            } else {
              return -0.00284894945f;
            }
          } else {
            return -0.00634907306f;
          }
        }
      } else {
        if (f[5] <= 2.5f) {
          if (f[2] <= 56746.5f) {
            if (f[0] <= 23.0f) {
              if (f[2] <= 32936.5f) {
                if (f[3] <= 14619.5f) {
                  return -0.00164022116f;
                } else {
                  if (f[4] <= 1.36606848f) {
                    return 0.00148842388f;
                  } else {
                    return 0.00382925667f;
                  }
                }
              } else {
                return 0.000215900737f;
              }
            } else {
              return 0.00273285379f;
            }
          } else {
            if (f[0] <= 21.0f) {
              return -0.00312083652f;
            } else {
              return 0.000106228171f;
            }
          }
        } else {
          if (f[5] <= 9.5f) {
            if (f[3] <= 69498.5f) {
              return 0.00286685077f;
            } else {
              return 0.000103974386f;
            }
          } else {
            return 0.00453377183f;
          }
        }
      }
    }
    case 25: {
      if (f[0] <= 13.0f) {
        if (f[0] <= 7.0f) {
          if (f[2] <= 37400.5f) {
            if (f[0] <= 3.0f) {
              return -0.00533895997f;
            } else {
              return -0.0024939431f;
            }
          } else {
            if (f[2] <= 67625.5f) {
              return -0.00518469512f;
            } else {
              return -0.00790989528f;
            }
          }
        } else {
          if (f[2] <= 56746.5f) {
            if (f[2] <= 32738.0f) {
              if (f[2] <= 7262.0f) {
                return -0.00389725308f;
              } else {
                return 0.000699374182f;
              }
            } else {
              return -0.00170194749f;
            }
          } else {
            return -0.00514674162f;
          }
        }
      } else {
        if (f[2] <= 41385.0f) {
          if (f[3] <= 13967.5f) {
            if (f[3] <= 3015.5f) {
              return -0.00750993093f;
            } else {
              return 0.00033753951f;
            }
          } else {
            if (f[0] <= 23.0f) {
              return 0.00229154127f;
            } else {
              return 0.00383650857f;
            }
          }
        } else {
          if (f[5] <= 1.00000002e-35f) {
            return -0.000536248371f;
          } else {
            if (f[2] <= 65602.0f) {
              return 0.00174833628f;
            } else {
              return -0.00161318132f;
            }
          }
        }
      }
    }
    case 26: {
      if (f[3] <= 71113.5f) {
        if (f[0] <= 15.0f) {
          if (f[3] <= 9632.0f) {
            return -0.00469184866f;
          } else {
            if (f[2] <= 32385.5f) {
              return 0.0010505435f;
            } else {
              return -0.000998413709f;
            }
          }
        } else {
          if (f[0] <= 27.0f) {
            if (f[2] <= 36591.0f) {
              if (f[3] <= 13967.5f) {
                return -0.000424707677f;
              } else {
                return 0.00289052839f;
              }
            } else {
              return 0.000973122284f;
            }
          } else {
            return 0.00318211773f;
          }
        }
      } else {
        if (f[0] <= 7.0f) {
          if (f[2] <= 37400.5f) {
            if (f[0] <= 3.0f) {
              return -0.00491184305f;
            } else {
              return -0.00229442764f;
            }
          } else {
            return -0.00519479656f;
          }
        } else {
          if (f[2] <= 79786.5f) {
            if (f[0] <= 27.0f) {
              if (f[2] <= 64275.5f) {
                return -0.00127493131f;
              } else {
                if (f[4] <= 1.11056352f) {
                  return -0.00806209963f;
                } else {
                  return -0.00228036229f;
                }
              }
            } else {
              return 0.0014544649f;
            }
          } else {
            return -0.012563808f;
          }
        }
      }
    }
    case 27: {
      if (f[0] <= 13.0f) {
        if (f[0] <= 7.0f) {
          if (f[2] <= 59535.5f) {
            if (f[0] <= 3.0f) {
              return -0.00499721474f;
            } else {
              return -0.00281787051f;
            }
          } else {
            return -0.00603039466f;
          }
        } else {
          if (f[2] <= 49930.5f) {
            if (f[4] <= 1.70894396f) {
              return -0.000752193943f;
            } else {
              return 0.00150824135f;
            }
          } else {
            if (f[2] <= 70062.5f) {
              return -0.00299213226f;
            } else {
              return -0.00842909322f;
            }
          }
        }
      } else {
        if (f[5] <= 2.5f) {
          if (f[2] <= 56746.5f) {
            if (f[3] <= 7991.5f) {
              return -0.00271442227f;
            } else {
              if (f[2] <= 32738.0f) {
                return 0.00229124851f;
              } else {
                return 0.000805932274f;
              }
            }
          } else {
            return -0.00102831796f;
          }
        } else {
          if (f[5] <= 10.5f) {
            if (f[3] <= 54993.0f) {
              if (f[3] <= 5226.0f) {
                return -0.00269398475f;
              } else {
                return 0.00266213689f;
              }
            } else {
              return 0.00131509623f;
            }
          } else {
            return 0.00396349537f;
          }
        }
      }
    }
    case 28: {
      if (f[0] <= 11.0f) {
        if (f[0] <= 5.0f) {
          if (f[2] <= 34406.5f) {
            return -0.00293595308f;
          } else {
            return -0.00473556606f;
          }
        } else {
          if (f[2] <= 42472.0f) {
            return -0.000884940091f;
          } else {
            if (f[4] <= 1.44255745f) {
              if (f[2] <= 74141.0f) {
                return -0.0033679383f;
              } else {
                return -0.00958234849f;
              }
            } else {
              return 0.000169603884f;
            }
          }
        }
      } else {
        if (f[5] <= 1.00000002e-35f) {
          if (f[2] <= 56746.5f) {
            if (f[2] <= 32936.5f) {
              if (f[3] <= 7991.5f) {
                return -0.00332978472f;
              } else {
                return 0.00174372318f;
              }
            } else {
              return -0.000119698003f;
            }
          } else {
            return -0.00236213785f;
          }
        } else {
          if (f[5] <= 9.5f) {
            if (f[2] <= 49930.5f) {
              if (f[3] <= 11209.5f) {
                return -0.000854340023f;
              } else {
                if (f[4] <= 1.41431701f) {
                  return 0.00181205165f;
                } else {
                  return 0.00336091946f;
                }
              }
            } else {
              return 0.000601376714f;
            }
          } else {
            return 0.00342579969f;
          }
        }
      }
    }
    case 29: {
      if (f[0] <= 13.0f) {
        if (f[0] <= 7.0f) {
          if (f[0] <= 3.0f) {
            return -0.00463005099f;
          } else {
            if (f[2] <= 72390.5f) {
              return -0.00255095379f;
            } else {
              return -0.00772393902f;
            }
          }
        } else {
          if (f[2] <= 58406.5f) {
            if (f[2] <= 32385.5f) {
              if (f[2] <= 6678.5f) {
                return -0.00356964081f;
              } else {
                return 0.000656776029f;
              }
            } else {
              return -0.00131975323f;
            }
          } else {
            if (f[4] <= 1.22994548f) {
              return -0.00605276617f;
            } else {
              return -0.00188127816f;
            }
          }
        }
      } else {
        if (f[0] <= 23.0f) {
          if (f[2] <= 56746.5f) {
            if (f[3] <= 3015.5f) {
              return -0.00784362506f;
            } else {
              if (f[4] <= 1.41767049f) {
                return 0.000755126425f;
              } else {
                return 0.00231761161f;
              }
            }
          } else {
            return -0.00149118665f;
          }
        } else {
          if (f[2] <= 42666.5f) {
            if (f[3] <= 7187.0f) {
              return -0.00146994567f;
            } else {
              return 0.00277670766f;
            }
          } else {
            return 0.00141446671f;
          }
        }
      }
    }
    case 30: {
      if (f[3] <= 69498.5f) {
        if (f[0] <= 19.0f) {
          if (f[3] <= 12615.5f) {
            return -0.00240477452f;
          } else {
            if (f[2] <= 32385.5f) {
              if (f[4] <= 1.40385002f) {
                return 0.000407226753f;
              } else {
                return 0.00236164801f;
              }
            } else {
              if (f[5] <= 1.00000002e-35f) {
                return -0.000881261912f;
              } else {
                return 0.000587330043f;
              }
            }
          }
        } else {
          if (f[3] <= 5226.0f) {
            return -0.00245842055f;
          } else {
            if (f[2] <= 35986.5f) {
              return 0.00249480325f;
            } else {
              return 0.00142010801f;
            }
          }
        }
      } else {
        if (f[0] <= 5.0f) {
          if (f[2] <= 33773.5f) {
            return -0.00239952535f;
          } else {
            return -0.00403976116f;
          }
        } else {
          if (f[2] <= 79786.5f) {
            if (f[0] <= 27.0f) {
              if (f[2] <= 31328.5f) {
                return -0.000275152628f;
              } else {
                if (f[4] <= 1.197097f) {
                  return -0.00256011462f;
                } else {
                  return -0.000625998358f;
                }
              }
            } else {
              return 0.00111427645f;
            }
          } else {
            return -0.0103679689f;
          }
        }
      }
    }
    case 31: {
      if (f[0] <= 13.0f) {
        if (f[2] <= 54630.0f) {
          if (f[0] <= 7.0f) {
            if (f[0] <= 3.0f) {
              return -0.00354226681f;
            } else {
              return -0.00188290949f;
            }
          } else {
            if (f[4] <= 1.6455195f) {
              return -0.000732634643f;
            } else {
              return 0.00112049047f;
            }
          }
        } else {
          if (f[4] <= 1.30134702f) {
            if (f[2] <= 74141.0f) {
              return -0.00379404551f;
            } else {
              return -0.0064448403f;
            }
          } else {
            return -0.00072602273f;
          }
        }
      } else {
        if (f[0] <= 25.0f) {
          if (f[2] <= 46340.5f) {
            if (f[3] <= 13967.5f) {
              return -0.000888008732f;
            } else {
              if (f[2] <= 31328.5f) {
                if (f[4] <= 1.33673155f) {
                  return 0.00104444669f;
                } else {
                  return 0.00313598666f;
                }
              } else {
                return 0.00078747439f;
              }
            }
          } else {
            return -0.000258689563f;
          }
        } else {
          if (f[4] <= 1.09946948f) {
            if (f[2] <= 65602.0f) {
              return 0.000852397212f;
            } else {
              return -0.00922901407f;
            }
          } else {
            return 0.00208467813f;
          }
        }
      }
    }
    case 32: {
      if (f[0] <= 13.0f) {
        if (f[2] <= 58406.5f) {
          if (f[0] <= 7.0f) {
            if (f[0] <= 3.0f) {
              return -0.00332975691f;
            } else {
              return -0.00178739105f;
            }
          } else {
            if (f[4] <= 1.51635349f) {
              return -0.000865468571f;
            } else {
              return 0.000698247324f;
            }
          }
        } else {
          if (f[2] <= 76519.0f) {
            return -0.00346751642f;
          } else {
            return -0.00626728946f;
          }
        }
      } else {
        if (f[5] <= 4.5f) {
          if (f[2] <= 53805.0f) {
            if (f[3] <= 3015.5f) {
              return -0.00637528032f;
            } else {
              if (f[0] <= 27.0f) {
                if (f[4] <= 1.424564f) {
                  return 0.000596155871f;
                } else {
                  return 0.00189382098f;
                }
              } else {
                return 0.00234915941f;
              }
            }
          } else {
            if (f[2] <= 65602.0f) {
              return -6.70664692e-05f;
            } else {
              if (f[4] <= 1.11056352f) {
                return -0.00827149252f;
              } else {
                return -0.000601133514f;
              }
            }
          }
        } else {
          if (f[5] <= 12.5f) {
            return 0.00171285733f;
          } else {
            return 0.00345387805f;
          }
        }
      }
    }
    case 33: {
      if (f[0] <= 11.0f) {
        if (f[2] <= 40482.5f) {
          if (f[0] <= 3.0f) {
            return -0.00277298994f;
          } else {
            return -0.000700113264f;
          }
        } else {
          if (f[4] <= 1.43874347f) {
            if (f[2] <= 71090.5f) {
              return -0.00267054886f;
            } else {
              return -0.00492803503f;
            }
          } else {
            return 0.000462430634f;
          }
        }
      } else {
        if (f[5] <= 1.00000002e-35f) {
          if (f[2] <= 50336.5f) {
            return 0.000448258693f;
          } else {
            return -0.00128642783f;
          }
        } else {
          if (f[5] <= 10.5f) {
            if (f[4] <= 1.15480798f) {
              if (f[2] <= 69231.0f) {
                return 0.000337492548f;
              } else {
                if (f[4] <= 1.11279249f) {
                  return -0.0105050489f;
                } else {
                  return 0.00023301622f;
                }
              }
            } else {
              if (f[3] <= 11209.5f) {
                return -0.000850672943f;
              } else {
                if (f[2] <= 31328.5f) {
                  if (f[4] <= 1.36148f) {
                    return 0.00143984287f;
                  } else {
                    return 0.00293502052f;
                  }
                } else {
                  return 0.00107695608f;
                }
              }
            }
          } else {
            return 0.00265693419f;
          }
        }
      }
    }
    case 34: {
      if (f[0] <= 15.0f) {
        if (f[4] <= 1.2239235f) {
          if (f[2] <= 56746.5f) {
            if (f[0] <= 3.0f) {
              return -0.00280472328f;
            } else {
              if (f[2] <= 32385.5f) {
                return -0.000434245007f;
              } else {
                return -0.00161177073f;
              }
            }
          } else {
            return -0.0035324147f;
          }
        } else {
          if (f[3] <= 9632.0f) {
            return -0.00370749824f;
          } else {
            if (f[2] <= 29809.5f) {
              return 0.000885493252f;
            } else {
              return -0.000490644434f;
            }
          }
        }
      } else {
        if (f[3] <= 54596.5f) {
          if (f[3] <= 14619.5f) {
            if (f[3] <= 3015.5f) {
              return -0.00538287102f;
            } else {
              return 7.82974058e-05f;
            }
          } else {
            if (f[4] <= 1.48282051f) {
              return 0.00146228485f;
            } else {
              return 0.00325422301f;
            }
          }
        } else {
          if (f[4] <= 1.09946948f) {
            if (f[3] <= 71845.5f) {
              return -0.00027468148f;
            } else {
              return -0.00897140865f;
            }
          } else {
            if (f[0] <= 27.0f) {
              return 0.000143296283f;
            } else {
              return 0.00147240127f;
            }
          }
        }
      }
    }
    case 35: {
      if (f[0] <= 13.0f) {
        if (f[2] <= 48841.5f) {
          if (f[0] <= 7.0f) {
            return -0.00161509016f;
          } else {
            if (f[2] <= 6138.5f) {
              return -0.00317785311f;
            } else {
              if (f[4] <= 1.70894396f) {
                return -0.000251734155f;
              } else {
                return 0.00169288008f;
              }
            }
          }
        } else {
          if (f[2] <= 74141.0f) {
            if (f[4] <= 1.30485398f) {
              return -0.00250954174f;
            } else {
              return -0.000570703192f;
            }
          } else {
            return -0.00482705551f;
          }
        }
      } else {
        if (f[0] <= 23.0f) {
          if (f[4] <= 1.16472352f) {
            if (f[3] <= 72568.5f) {
              return -0.000431901423f;
            } else {
              return -0.00450102484f;
            }
          } else {
            if (f[3] <= 3015.5f) {
              return -0.0060667551f;
            } else {
              return 0.000688343072f;
            }
          }
        } else {
          if (f[5] <= 12.5f) {
            if (f[4] <= 1.10273802f) {
              if (f[3] <= 71845.5f) {
                return 0.000238707923f;
              } else {
                return -0.00783952252f;
              }
            } else {
              return 0.00134206458f;
            }
          } else {
            return 0.0028414061f;
          }
        }
      }
    }
    case 36: {
      if (f[3] <= 69498.5f) {
        if (f[0] <= 19.0f) {
          if (f[3] <= 14619.5f) {
            return -0.0017887517f;
          } else {
            if (f[2] <= 31512.5f) {
              if (f[4] <= 1.35927302f) {
                return -1.97258817e-05f;
              } else {
                return 0.00179870668f;
              }
            } else {
              if (f[5] <= 1.00000002e-35f) {
                return -0.000723953754f;
              } else {
                return 0.000381921338f;
              }
            }
          }
        } else {
          if (f[3] <= 5226.0f) {
            return -0.00215851225f;
          } else {
            if (f[4] <= 1.45025247f) {
              return 0.00110542261f;
            } else {
              return 0.00242998418f;
            }
          }
        }
      } else {
        if (f[0] <= 3.0f) {
          return -0.00270664421f;
        } else {
          if (f[2] <= 79786.5f) {
            if (f[4] <= 1.22027302f) {
              if (f[2] <= 31129.5f) {
                return -0.000165507405f;
              } else {
                return -0.00161187042f;
              }
            } else {
              if (f[0] <= 27.0f) {
                return -0.000349957249f;
              } else {
                if (f[2] <= 63871.5f) {
                  return 0.0017319795f;
                } else {
                  return 0.0103584417f;
                }
              }
            }
          } else {
            return -0.00721900837f;
          }
        }
      }
    }
    case 37: {
      if (f[0] <= 15.0f) {
        if (f[2] <= 59270.0f) {
          if (f[0] <= 7.0f) {
            return -0.00152428848f;
          } else {
            if (f[3] <= 9632.0f) {
              return -0.00322101748f;
            } else {
              if (f[2] <= 31129.5f) {
                return 0.000510280854f;
              } else {
                return -0.000580269031f;
              }
            }
          }
        } else {
          return -0.00272694987f;
        }
      } else {
        if (f[2] <= 36591.0f) {
          if (f[3] <= 14619.5f) {
            if (f[2] <= 1754.5f) {
              return -0.00467907886f;
            } else {
              return -3.97499304e-05f;
            }
          } else {
            if (f[4] <= 1.36352849f) {
              return 0.00122769352f;
            } else {
              if (f[2] <= 31512.5f) {
                return 0.00269587572f;
              } else {
                return 0.00060954352f;
              }
            }
          }
        } else {
          if (f[0] <= 27.0f) {
            if (f[4] <= 1.15600252f) {
              if (f[3] <= 72568.5f) {
                return -0.000416755075f;
              } else {
                if (f[4] <= 1.11056352f) {
                  return -0.00777688309f;
                } else {
                  return -0.00103314909f;
                }
              }
            } else {
              return 0.000415334372f;
            }
          } else {
            return 0.00112287874f;
          }
        }
      }
    }
    case 38: {
      if (f[0] <= 11.0f) {
        if (f[2] <= 40482.5f) {
          if (f[3] <= 12615.5f) {
            return -0.00516150539f;
          } else {
            if (f[4] <= 1.98430747f) {
              if (f[0] <= 3.0f) {
                return -0.00185909102f;
              } else {
                return -0.000484358324f;
              }
            } else {
              return 0.00274770088f;
            }
          }
        } else {
          if (f[4] <= 1.43874347f) {
            return -0.00202911694f;
          } else {
            return 0.000586575417f;
          }
        }
      } else {
        if (f[5] <= 1.00000002e-35f) {
          if (f[2] <= 68329.5f) {
            if (f[2] <= 32936.5f) {
              if (f[3] <= 15987.0f) {
                return -0.00109017751f;
              } else {
                if (f[4] <= 1.33891648f) {
                  return 0.000312889754f;
                } else {
                  return 0.00216955903f;
                }
              }
            } else {
              return -0.000363707185f;
            }
          } else {
            return -0.00386763045f;
          }
        } else {
          if (f[4] <= 1.32443148f) {
            if (f[0] <= 21.0f) {
              return 0.000147938403f;
            } else {
              return 0.0008955877f;
            }
          } else {
            if (f[3] <= 3015.5f) {
              return -0.00407930098f;
            } else {
              return 0.00149686117f;
            }
          }
        }
      }
    }
    case 39: {
      if (f[0] <= 15.0f) {
        if (f[2] <= 59270.0f) {
          if (f[0] <= 5.0f) {
            return -0.00156192904f;
          } else {
            if (f[4] <= 1.51635349f) {
              if (f[2] <= 6678.5f) {
                return -0.00304829515f;
              } else {
                if (f[2] <= 28849.0f) {
                  return 0.000223400944f;
                } else {
                  return -0.00069718094f;
                }
              }
            } else {
              return 0.000637020358f;
            }
          }
        } else {
          return -0.00236722133f;
        }
      } else {
        if (f[3] <= 54596.5f) {
          if (f[3] <= 14619.5f) {
            return -0.00035867303f;
          } else {
            if (f[4] <= 1.48282051f) {
              if (f[0] <= 27.0f) {
                return 0.000790571468f;
              } else {
                return 0.00166330836f;
              }
            } else {
              return 0.00240063107f;
            }
          }
        } else {
          if (f[4] <= 1.09946948f) {
            if (f[3] <= 71845.5f) {
              if (f[2] <= 60201.5f) {
                return -0.00195441818f;
              } else {
                return 0.00283741207f;
              }
            } else {
              return -0.00720968618f;
            }
          } else {
            if (f[0] <= 27.0f) {
              return 1.71571396e-05f;
            } else {
              return 0.00102041271f;
            }
          }
        }
      }
    }
    case 40: {
      if (f[0] <= 11.0f) {
        if (f[2] <= 35798.5f) {
          if (f[3] <= 12615.5f) {
            return -0.00475276422f;
          } else {
            if (f[4] <= 1.98430747f) {
              return -0.000484954091f;
            } else {
              return 0.00264367659f;
            }
          }
        } else {
          if (f[4] <= 1.37642998f) {
            return -0.00169169857f;
          } else {
            return 7.11904961e-05f;
          }
        }
      } else {
        if (f[0] <= 21.0f) {
          if (f[4] <= 1.16085547f) {
            if (f[3] <= 72568.5f) {
              return -0.000851007354f;
            } else {
              return -0.0048597921f;
            }
          } else {
            if (f[3] <= 3015.5f) {
              return -0.0054884367f;
            } else {
              if (f[4] <= 1.3569535f) {
                return 5.40855277e-05f;
              } else {
                if (f[2] <= 31512.5f) {
                  return 0.00153831647f;
                } else {
                  return -0.000106018448f;
                }
              }
            }
          }
        } else {
          if (f[5] <= 12.5f) {
            if (f[4] <= 1.15480798f) {
              return 8.48430382e-05f;
            } else {
              if (f[3] <= 5226.0f) {
                return -0.00211114828f;
              } else {
                return 0.00102437997f;
              }
            }
          } else {
            return 0.00220160828f;
          }
        }
      }
    }
    case 41: {
      if (f[0] <= 15.0f) {
        if (f[2] <= 59270.0f) {
          if (f[0] <= 5.0f) {
            return -0.00134656878f;
          } else {
            if (f[2] <= 6678.5f) {
              return -0.00217588874f;
            } else {
              if (f[2] <= 28849.0f) {
                if (f[3] <= 31686.5f) {
                  return -0.000379237522f;
                } else {
                  return 0.000960840488f;
                }
              } else {
                return -0.000501947125f;
              }
            }
          }
        } else {
          if (f[2] <= 76519.0f) {
            return -0.0017227796f;
          } else {
            return -0.00385080064f;
          }
        }
      } else {
        if (f[3] <= 43181.0f) {
          if (f[3] <= 23198.5f) {
            if (f[2] <= 1754.5f) {
              return -0.00386128873f;
            } else {
              return 0.000262304086f;
            }
          } else {
            if (f[4] <= 1.36606848f) {
              return 0.0011970238f;
            } else {
              return 0.00244652025f;
            }
          }
        } else {
          if (f[0] <= 27.0f) {
            if (f[5] <= 4.5f) {
              if (f[4] <= 1.16085547f) {
                return -0.000971410679f;
              } else {
                return 0.000139941024f;
              }
            } else {
              return 0.000877610586f;
            }
          } else {
            return 0.000866895247f;
          }
        }
      }
    }
    case 42: {
      if (f[5] <= 1.00000002e-35f) {
        if (f[2] <= 68329.5f) {
          if (f[0] <= 7.0f) {
            if (f[0] <= 3.0f) {
              return -0.00173180661f;
            } else {
              return -0.000798239103f;
            }
          } else {
            if (f[2] <= 36591.0f) {
              if (f[2] <= 5473.5f) {
                return -0.00246699844f;
              } else {
                return 0.000451050252f;
              }
            } else {
              if (f[4] <= 2.4659369f) {
                return -0.000476708415f;
              } else {
                return -0.0127129323f;
              }
            }
          }
        } else {
          return -0.00266245997f;
        }
      } else {
        if (f[5] <= 11.5f) {
          if (f[4] <= 1.15480798f) {
            if (f[2] <= 69231.0f) {
              return 6.13611026e-05f;
            } else {
              if (f[4] <= 1.11279249f) {
                return -0.00782775978f;
              } else {
                if (f[2] <= 72390.5f) {
                  return -0.00358371461f;
                } else {
                  return 0.00612831298f;
                }
              }
            }
          } else {
            if (f[3] <= 19434.5f) {
              return -0.000216443655f;
            } else {
              if (f[2] <= 30473.5f) {
                return 0.0014750592f;
              } else {
                return 0.000568040234f;
              }
            }
          }
        } else {
          return 0.00173921196f;
        }
      }
    }
    case 43: {
      if (f[0] <= 15.0f) {
        if (f[2] <= 49930.5f) {
          if (f[0] <= 7.0f) {
            return -0.000850182344f;
          } else {
            if (f[3] <= 15987.0f) {
              return -0.00179291251f;
            } else {
              if (f[4] <= 1.70894396f) {
                return -4.04351468e-05f;
              } else {
                return 0.00136735218f;
              }
            }
          }
        } else {
          if (f[2] <= 76519.0f) {
            if (f[4] <= 1.25408155f) {
              return -0.00145835604f;
            } else {
              return -0.000155929984f;
            }
          } else {
            return -0.00333119847f;
          }
        }
      } else {
        if (f[3] <= 42978.5f) {
          if (f[3] <= 23198.5f) {
            return 0.000106411296f;
          } else {
            if (f[4] <= 1.36606848f) {
              return 0.00104319824f;
            } else {
              return 0.00214695486f;
            }
          }
        } else {
          if (f[0] <= 27.0f) {
            if (f[5] <= 5.5f) {
              if (f[4] <= 1.16472352f) {
                if (f[2] <= 56746.5f) {
                  return -0.000213052752f;
                } else {
                  return -0.00182765303f;
                }
              } else {
                return 0.000140491732f;
              }
            } else {
              return 0.000944003664f;
            }
          } else {
            return 0.000765317606f;
          }
        }
      }
    }
    case 44: {
      if (f[5] <= 1.00000002e-35f) {
        if (f[2] <= 68329.5f) {
          if (f[0] <= 3.0f) {
            return -0.00150805098f;
          } else {
            if (f[2] <= 31129.5f) {
              if (f[2] <= 5473.5f) {
                return -0.00208445443f;
              } else {
                if (f[4] <= 1.48282051f) {
                  return 0.000171856939f;
                } else {
                  return 0.00126443175f;
                }
              }
            } else {
              if (f[0] <= 27.0f) {
                if (f[4] <= 2.4659369f) {
                  if (f[4] <= 1.19600546f) {
                    return -0.000860902833f;
                  } else {
                    if (f[3] <= 82127.5f) {
                      return -0.000383017881f;
                    } else {
                      return 0.00162024223f;
                    }
                  }
                } else {
                  return -0.00862070821f;
                }
              } else {
                return 0.00110520478f;
              }
            }
          }
        } else {
          return -0.00229372126f;
        }
      } else {
        if (f[5] <= 13.5f) {
          if (f[4] <= 1.15480798f) {
            if (f[3] <= 65011.5f) {
              return 0.000257179681f;
            } else {
              return -0.00110668791f;
            }
          } else {
            if (f[3] <= 5226.0f) {
              return -0.00174114493f;
            } else {
              return 0.000671581716f;
            }
          }
        } else {
          return 0.00198537258f;
        }
      }
    }
    case 45: {
      if (f[0] <= 11.0f) {
        if (f[2] <= 28579.0f) {
          if (f[3] <= 21387.5f) {
            return -0.00266740385f;
          } else {
            return 4.16793828e-05f;
          }
        } else {
          if (f[4] <= 1.46839052f) {
            return -0.00110038162f;
          } else {
            return 0.000347765775f;
          }
        }
      } else {
        if (f[3] <= 69214.5f) {
          if (f[3] <= 3015.5f) {
            return -0.00376181047f;
          } else {
            if (f[4] <= 1.35927302f) {
              if (f[0] <= 19.0f) {
                if (f[2] <= 15184.5f) {
                  return -0.00174206016f;
                } else {
                  return -3.87102492e-05f;
                }
              } else {
                return 0.000532932583f;
              }
            } else {
              if (f[2] <= 31512.5f) {
                if (f[3] <= 13239.5f) {
                  return 0.000138664286f;
                } else {
                  return 0.00161365835f;
                }
              } else {
                if (f[2] <= 33532.5f) {
                  return -0.0013607677f;
                } else {
                  return 0.000648240462f;
                }
              }
            }
          }
        } else {
          if (f[4] <= 1.11056352f) {
            if (f[3] <= 71113.5f) {
              return 0.00120792126f;
            } else {
              return -0.00571296281f;
            }
          } else {
            return -0.000259457846f;
          }
        }
      }
    }
    case 46: {
      if (f[0] <= 15.0f) {
        if (f[2] <= 49930.5f) {
          if (f[0] <= 5.0f) {
            return -0.000855085f;
          } else {
            if (f[2] <= 7262.0f) {
              return -0.00156424123f;
            } else {
              return 1.85883104e-06f;
            }
          }
        } else {
          if (f[2] <= 76519.0f) {
            return -0.000967779917f;
          } else {
            return -0.00279518118f;
          }
        }
      } else {
        if (f[4] <= 1.22994548f) {
          if (f[3] <= 65444.5f) {
            if (f[3] <= 29870.5f) {
              return -0.000723267442f;
            } else {
              return 0.000499639496f;
            }
          } else {
            return -0.000529285212f;
          }
        } else {
          if (f[2] <= 31328.5f) {
            if (f[3] <= 24957.0f) {
              if (f[3] <= 3015.5f) {
                return -0.00321710095f;
              } else {
                return 0.000516379761f;
              }
            } else {
              return 0.00157238647f;
            }
          } else {
            if (f[2] <= 57799.0f) {
              if (f[4] <= 1.72799802f) {
                return 0.000240659015f;
              } else {
                return -0.00620324007f;
              }
            } else {
              if (f[2] <= 58406.5f) {
                return 0.00862269427f;
              } else {
                return 0.00142707499f;
              }
            }
          }
        }
      }
    }
    case 47: {
      if (f[5] <= 1.00000002e-35f) {
        if (f[2] <= 68329.5f) {
          if (f[0] <= 3.0f) {
            return -0.00125156666f;
          } else {
            if (f[2] <= 31129.5f) {
              if (f[2] <= 5473.5f) {
                return -0.00176672242f;
              } else {
                return 0.000354637017f;
              }
            } else {
              if (f[0] <= 27.0f) {
                if (f[4] <= 2.4659369f) {
                  return -0.000476967144f;
                } else {
                  return -0.00791817942f;
                }
              } else {
                return 0.0009767421f;
              }
            }
          }
        } else {
          return -0.00190137628f;
        }
      } else {
        if (f[5] <= 12.5f) {
          if (f[4] <= 1.15480798f) {
            return -9.26263419e-05f;
          } else {
            if (f[3] <= 18092.5f) {
              return -0.000277876321f;
            } else {
              if (f[2] <= 30473.5f) {
                if (f[4] <= 1.24331599f) {
                  return 0.000210584072f;
                } else {
                  if (f[3] <= 24957.0f) {
                    return 0.000381662056f;
                  } else {
                    return 0.0015376791f;
                  }
                }
              } else {
                if (f[3] <= 88860.5f) {
                  return 0.00038228015f;
                } else {
                  return 0.00629113285f;
                }
              }
            }
          }
        } else {
          return 0.00149970836f;
        }
      }
    }
    case 48: {
      if (f[5] <= 1.00000002e-35f) {
        if (f[4] <= 1.22994548f) {
          if (f[2] <= 56746.5f) {
            if (f[3] <= 12045.0f) {
              return -0.0050752782f;
            } else {
              if (f[0] <= 3.0f) {
                return -0.00105624854f;
              } else {
                return -0.000264130128f;
              }
            }
          } else {
            return -0.00131842154f;
          }
        } else {
          if (f[2] <= 1754.5f) {
            return -0.00511083546f;
          } else {
            if (f[0] <= 27.0f) {
              if (f[4] <= 1.48282051f) {
                return -0.000198907611f;
              } else {
                return 0.000522948434f;
              }
            } else {
              return 0.00207273017f;
            }
          }
        }
      } else {
        if (f[5] <= 13.5f) {
          if (f[4] <= 1.32443148f) {
            if (f[2] <= 18699.0f) {
              return -0.000612791303f;
            } else {
              if (f[2] <= 40654.5f) {
                return 0.000647938519f;
              } else {
                return 0.000120809121f;
              }
            }
          } else {
            if (f[2] <= 56061.5f) {
              if (f[3] <= 69214.5f) {
                return 0.000718907304f;
              } else {
                return -0.00168272382f;
              }
            } else {
              return 0.00567668817f;
            }
          }
        } else {
          return 0.00160250025f;
        }
      }
    }
    case 49: {
      if (f[0] <= 15.0f) {
        if (f[4] <= 1.41767049f) {
          if (f[2] <= 59270.0f) {
            if (f[3] <= 21904.0f) {
              return -0.00226860676f;
            } else {
              if (f[0] <= 3.0f) {
                return -0.00101436259f;
              } else {
                if (f[2] <= 40482.5f) {
                  return 3.05526276e-07f;
                } else {
                  return -0.000547240477f;
                }
              }
            }
          } else {
            return -0.00128825243f;
          }
        } else {
          return 0.000297572266f;
        }
      } else {
        if (f[0] <= 27.0f) {
          if (f[4] <= 1.15480798f) {
            if (f[2] <= 56746.5f) {
              if (f[3] <= 59329.5f) {
                if (f[2] <= 52381.0f) {
                  return -0.000237702077f;
                } else {
                  return -0.00446203693f;
                }
              } else {
                return 0.00154351674f;
              }
            } else {
              return -0.00139644622f;
            }
          } else {
            if (f[3] <= 56881.5f) {
              if (f[3] <= 19434.5f) {
                return -0.000271059991f;
              } else {
                return 0.000601664998f;
              }
            } else {
              if (f[4] <= 1.72799802f) {
                return -3.77921131e-05f;
              } else {
                return -0.00648934718f;
              }
            }
          }
        } else {
          return 0.000726847594f;
        }
      }
    }
    case 50: {
      if (f[0] <= 11.0f) {
        if (f[4] <= 1.6455195f) {
          if (f[3] <= 77840.5f) {
            return -0.00151775166f;
          } else {
            if (f[2] <= 28579.0f) {
              return 7.97734053e-05f;
            } else {
              return -0.000656480348f;
            }
          }
        } else {
          return 0.000674578739f;
        }
      } else {
        if (f[2] <= 79786.5f) {
          if (f[0] <= 27.0f) {
            if (f[4] <= 1.15480798f) {
              if (f[0] <= 13.0f) {
                if (f[3] <= 65011.5f) {
                  return -0.00162774883f;
                } else {
                  return -0.0074246821f;
                }
              } else {
                return -0.00040032936f;
              }
            } else {
              if (f[3] <= 3015.5f) {
                return -0.00333298533f;
              } else {
                if (f[3] <= 56881.5f) {
                  if (f[4] <= 1.48282051f) {
                    if (f[3] <= 18092.5f) {
                      return -0.00065928512f;
                    } else {
                      return 0.000382053872f;
                    }
                  } else {
                    return 0.00108045377f;
                  }
                } else {
                  if (f[4] <= 1.72799802f) {
                    return -2.7605294e-05f;
                  } else {
                    return -0.0035100258f;
                  }
                }
              }
            }
          } else {
            return 0.00067099784f;
          }
        } else {
          return -0.00619553684f;
        }
      }
    }
    case 51: {
      if (f[0] <= 19.0f) {
        if (f[4] <= 1.2239235f) {
          if (f[2] <= 40856.5f) {
            if (f[3] <= 31686.5f) {
              return -0.00161774451f;
            } else {
              if (f[0] <= 7.0f) {
                return -0.000411158122f;
              } else {
                return 0.000622152008f;
              }
            }
          } else {
            return -0.000837564154f;
          }
        } else {
          if (f[3] <= 3015.5f) {
            return -0.0041978378f;
          } else {
            if (f[0] <= 11.0f) {
              if (f[4] <= 1.48282051f) {
                if (f[2] <= 15184.5f) {
                  return -0.00587079149f;
                } else {
                  return -0.00104261104f;
                }
              } else {
                return 0.000344167024f;
              }
            } else {
              return 0.000279754462f;
            }
          }
        }
      } else {
        if (f[3] <= 42978.5f) {
          if (f[3] <= 29163.0f) {
            if (f[2] <= 24849.5f) {
              if (f[4] <= 1.2347315f) {
                return -0.000461428645f;
              } else {
                if (f[3] <= 7187.0f) {
                  return -0.000926276536f;
                } else {
                  return 0.000863070205f;
                }
              }
            } else {
              return -0.00550306669f;
            }
          } else {
            return 0.00108059506f;
          }
        } else {
          return 0.000204559406f;
        }
      }
    }
    case 52: {
      if (f[5] <= 1.00000002e-35f) {
        if (f[2] <= 68329.5f) {
          if (f[4] <= 1.25124145f) {
            if (f[3] <= 12045.0f) {
              return -0.00441429844f;
            } else {
              if (f[2] <= 40856.5f) {
                if (f[0] <= 7.0f) {
                  return -0.000378265473f;
                } else {
                  if (f[3] <= 31686.5f) {
                    return -0.000973099809f;
                  } else {
                    return 0.000528168303f;
                  }
                }
              } else {
                return -0.000610751363f;
              }
            }
          } else {
            if (f[2] <= 1754.5f) {
              return -0.00433637864f;
            } else {
              if (f[0] <= 11.0f) {
                if (f[4] <= 1.48282051f) {
                  if (f[2] <= 15184.5f) {
                    return -0.00532615357f;
                  } else {
                    return -0.000883519866f;
                  }
                } else {
                  return 0.000324141645f;
                }
              } else {
                if (f[2] <= 29130.0f) {
                  return 0.00100265156f;
                } else {
                  if (f[2] <= 57799.0f) {
                    return -0.000101404258f;
                  } else {
                    return 0.00188438144f;
                  }
                }
              }
            }
          }
        } else {
          return -0.00144715318f;
        }
      } else {
        if (f[5] <= 13.5f) {
          return 0.000277763731f;
        } else {
          return 0.00132477492f;
        }
      }
    }
    case 53: {
      if (f[0] <= 19.0f) {
        if (f[4] <= 1.291484f) {
          if (f[2] <= 56746.5f) {
            return -0.000262283149f;
          } else {
            return -0.000980344257f;
          }
        } else {
          if (f[3] <= 5226.0f) {
            if (f[4] <= 1.34971553f) {
              return -0.010177419f;
            } else {
              return -0.00150916501f;
            }
          } else {
            return 0.000198591683f;
          }
        }
      } else {
        if (f[4] <= 1.09946948f) {
          if (f[3] <= 71113.5f) {
            if (f[3] <= 64603.5f) {
              if (f[2] <= 58406.5f) {
                if (f[3] <= 28043.5f) {
                  if (f[2] <= 24849.5f) {
                    return -0.00300619104f;
                  } else {
                    return -0.0154304169f;
                  }
                } else {
                  return 0.000226740196f;
                }
              } else {
                if (f[2] <= 58999.5f) {
                  return -0.0150024768f;
                } else {
                  if (f[3] <= 64370.0f) {
                    return 0.000733247179f;
                  } else {
                    return -0.0132848867f;
                  }
                }
              }
            } else {
              return 0.00277209264f;
            }
          } else {
            return -0.004950938f;
          }
        } else {
          if (f[0] <= 29.0f) {
            return 0.000260182167f;
          } else {
            return 0.00074827982f;
          }
        }
      }
    }
    case 54: {
      if (f[0] <= 15.0f) {
        if (f[2] <= 71090.5f) {
          if (f[4] <= 1.51635349f) {
            if (f[3] <= 22406.0f) {
              if (f[0] <= 11.0f) {
                return -0.0043542822f;
              } else {
                return -0.000960935652f;
              }
            } else {
              if (f[1] <= 3.0f) {
                return -0.000814886319f;
              } else {
                if (f[2] <= 28849.0f) {
                  return 0.000214326385f;
                } else {
                  return -0.00031330439f;
                }
              }
            }
          } else {
            return 0.000405987299f;
          }
        } else {
          return -0.0014867137f;
        }
      } else {
        if (f[2] <= 85832.0f) {
          if (f[3] <= 42978.5f) {
            if (f[3] <= 30190.0f) {
              if (f[4] <= 1.2347315f) {
                return -0.000664293831f;
              } else {
                if (f[2] <= 1754.5f) {
                  if (f[4] <= 1.27853101f) {
                    return -0.0147377197f;
                  } else {
                    if (f[4] <= 1.33274752f) {
                      return 0.00861996874f;
                    } else {
                      return -0.00261942023f;
                    }
                  }
                } else {
                  return 0.000518543269f;
                }
              }
            } else {
              return 0.000923427979f;
            }
          } else {
            return 8.1005739e-05f;
          }
        } else {
          return -0.0125348286f;
        }
      }
    }
    case 55: {
      if (f[0] <= 21.0f) {
        if (f[4] <= 1.2239235f) {
          if (f[2] <= 56746.5f) {
            return -0.000226241871f;
          } else {
            if (f[4] <= 1.06721401f) {
              return -0.000584504073f;
            } else {
              if (f[0] <= 13.0f) {
                if (f[4] <= 1.120655f) {
                  return -0.0112131082f;
                } else {
                  return -0.00348356106f;
                }
              } else {
                if (f[2] <= 79786.5f) {
                  if (f[3] <= 74208.5f) {
                    return -0.00156342217f;
                  } else {
                    if (f[4] <= 1.12458497f) {
                      return -0.00470380422f;
                    } else {
                      return 0.00118465757f;
                    }
                  }
                } else {
                  return -0.0077580183f;
                }
              }
            }
          }
        } else {
          return 7.78331172e-05f;
        }
      } else {
        if (f[4] <= 1.10273802f) {
          if (f[2] <= 69231.0f) {
            if (f[3] <= 73758.5f) {
              if (f[3] <= 71845.5f) {
                return -0.000116385267f;
              } else {
                return -0.00936282652f;
              }
            } else {
              return 0.0103367965f;
            }
          } else {
            return -0.00639375823f;
          }
        } else {
          if (f[2] <= 76519.0f) {
            return 0.00036762368f;
          } else {
            return 0.00717829611f;
          }
        }
      }
    }
    case 56: {
      if (f[5] <= 1.00000002e-35f) {
        if (f[2] <= 28579.0f) {
          if (f[2] <= 5473.5f) {
            return -0.00133957791f;
          } else {
            return 0.00029485308f;
          }
        } else {
          if (f[0] <= 27.0f) {
            if (f[2] <= 68329.5f) {
              if (f[4] <= 2.4659369f) {
                return -0.000325243431f;
              } else {
                if (f[2] <= 42666.5f) {
                  return -0.00188696615f;
                } else {
                  return -0.0135258428f;
                }
              }
            } else {
              return -0.00122092153f;
            }
          } else {
            return 0.000831116795f;
          }
        }
      } else {
        if (f[4] <= 2.4659369f) {
          if (f[3] <= 88860.5f) {
            if (f[2] <= 76519.0f) {
              if (f[3] <= 19434.5f) {
                return -0.000306908558f;
              } else {
                if (f[3] <= 43181.0f) {
                  return 0.000585590185f;
                } else {
                  if (f[0] <= 33.0f) {
                    return 0.000169218171f;
                  } else {
                    return -0.00116475481f;
                  }
                }
              }
            } else {
              return -0.00693556207f;
            }
          } else {
            if (f[4] <= 1.1327765f) {
              return -0.00781567797f;
            } else {
              return 0.00655174146f;
            }
          }
        } else {
          return 0.00448406874f;
        }
      }
    }
    case 57: {
      if (f[0] <= 21.0f) {
        if (f[2] <= 76519.0f) {
          if (f[4] <= 1.48282051f) {
            if (f[3] <= 19903.5f) {
              if (f[2] <= 16699.5f) {
                if (f[0] <= 11.0f) {
                  return -0.00432473239f;
                } else {
                  return -0.000821831043f;
                }
              } else {
                return -0.0117406882f;
              }
            } else {
              if (f[0] <= 3.0f) {
                return -0.000700901854f;
              } else {
                if (f[2] <= 28849.0f) {
                  return 0.00023384297f;
                } else {
                  if (f[2] <= 29355.5f) {
                    if (f[3] <= 35642.5f) {
                      if (f[4] <= 1.17423302f) {
                        return 0.000429456406f;
                      } else {
                        return -0.012829568f;
                      }
                    } else {
                      return -0.000867061604f;
                    }
                  } else {
                    if (f[3] <= 38574.5f) {
                      return 0.00178328521f;
                    } else {
                      return -0.000223450173f;
                    }
                  }
                }
              }
            }
          } else {
            if (f[3] <= 3015.5f) {
              return -0.00358932129f;
            } else {
              return 0.000457582144f;
            }
          }
        } else {
          if (f[0] <= 3.0f) {
            return -0.000253916229f;
          } else {
            return -0.00326780738f;
          }
        }
      } else {
        return 0.000287276646f;
      }
    }
    case 58: {
      if (f[4] <= 1.16085547f) {
        if (f[2] <= 1.00000002e-35f) {
          return 0.0106992094f;
        } else {
          if (f[2] <= 56746.5f) {
            return -0.000135991894f;
          } else {
            if (f[3] <= 63772.5f) {
              if (f[2] <= 58406.5f) {
                if (f[2] <= 57239.5f) {
                  return -0.00626894266f;
                } else {
                  return 0.00494702726f;
                }
              } else {
                return -0.0126304513f;
              }
            } else {
              return -0.000614954531f;
            }
          }
        }
      } else {
        if (f[0] <= 11.0f) {
          if (f[4] <= 1.30485398f) {
            return -0.00246763336f;
          } else {
            if (f[3] <= 82127.5f) {
              if (f[4] <= 1.98430747f) {
                return -0.000666804884f;
              } else {
                return 0.00171793117f;
              }
            } else {
              return 0.00148274615f;
            }
          }
        } else {
          if (f[3] <= 5226.0f) {
            if (f[4] <= 1.42111003f) {
              return -0.0032478196f;
            } else {
              if (f[3] <= 3015.5f) {
                if (f[4] <= 1.48732197f) {
                  return 0.00341345871f;
                } else {
                  return -0.00409455977f;
                }
              } else {
                return 0.0018654791f;
              }
            }
          } else {
            return 0.00025856081f;
          }
        }
      }
    }
    case 59: {
      if (f[0] <= 23.0f) {
        if (f[5] <= 4.5f) {
          if (f[2] <= 76519.0f) {
            if (f[4] <= 1.48282051f) {
              if (f[3] <= 19903.5f) {
                if (f[0] <= 11.0f) {
                  return -0.00410506116f;
                } else {
                  return -0.000791627026f;
                }
              } else {
                if (f[2] <= 28579.0f) {
                  return 0.00014002474f;
                } else {
                  return -0.000252796301f;
                }
              }
            } else {
              return 0.00030822192f;
            }
          } else {
            if (f[0] <= 5.0f) {
              return -0.000597639275f;
            } else {
              return -0.00365410681f;
            }
          }
        } else {
          if (f[3] <= 82127.5f) {
            if (f[4] <= 1.07681853f) {
              return 0.00968934661f;
            } else {
              return 0.00051961675f;
            }
          } else {
            return 0.013373627f;
          }
        }
      } else {
        if (f[4] <= 1.0832895f) {
          if (f[3] <= 54400.5f) {
            return 0.00108171007f;
          } else {
            if (f[5] <= 4.5f) {
              return -0.00469922569f;
            } else {
              if (f[2] <= 54887.0f) {
                return -0.00967195995f;
              } else {
                return 0.00243250818f;
              }
            }
          }
        } else {
          return 0.000329622938f;
        }
      }
    }
    case 60: {
      if (f[4] <= 1.16085547f) {
        if (f[2] <= 1.00000002e-35f) {
          return 0.0097033191f;
        } else {
          if (f[2] <= 56746.5f) {
            return -0.000121259804f;
          } else {
            if (f[3] <= 63772.5f) {
              if (f[2] <= 58406.5f) {
                if (f[2] <= 57239.5f) {
                  return -0.00573678672f;
                } else {
                  return 0.00457719705f;
                }
              } else {
                return -0.011435105f;
              }
            } else {
              return -0.000541734719f;
            }
          }
        }
      } else {
        if (f[0] <= 11.0f) {
          if (f[4] <= 1.30485398f) {
            return -0.00223846944f;
          } else {
            if (f[3] <= 79325.5f) {
              if (f[4] <= 2.07034707f) {
                if (f[2] <= 6678.5f) {
                  return -0.00399746523f;
                } else {
                  return -0.00050098213f;
                }
              } else {
                return 0.00180885834f;
              }
            } else {
              return 0.00119909436f;
            }
          }
        } else {
          if (f[3] <= 5226.0f) {
            if (f[4] <= 1.42111003f) {
              return -0.0029713865f;
            } else {
              if (f[4] <= 1.561248f) {
                return 0.00268512476f;
              } else {
                return -0.00150052739f;
              }
            }
          } else {
            return 0.00023150814f;
          }
        }
      }
    }
    case 61: {
      if (f[0] <= 27.0f) {
        if (f[4] <= 1.22994548f) {
          if (f[3] <= 31328.5f) {
            return -0.00102397056f;
          } else {
            if (f[2] <= 56746.5f) {
              if (f[0] <= 7.0f) {
                return -0.000326812793f;
              } else {
                if (f[2] <= 56256.5f) {
                  if (f[2] <= 40856.5f) {
                    return 0.000453873733f;
                  } else {
                    if (f[3] <= 50691.5f) {
                      if (f[2] <= 45204.5f) {
                        if (f[4] <= 1.16472352f) {
                          if (f[4] <= 1.15958703f) {
                            if (f[3] <= 49884.0f) {
                              return -0.00186443881f;
                            } else {
                              return -0.00666206784f;
                            }
                          } else {
                            return -0.0108331773f;
                          }
                        } else {
                          return 1.27293483e-05f;
                        }
                      } else {
                        return 0.00541626486f;
                      }
                    } else {
                      return 5.992202e-05f;
                    }
                  }
                } else {
                  if (f[3] <= 64820.0f) {
                    return 0.0054683839f;
                  } else {
                    return 0.00082153753f;
                  }
                }
              }
            } else {
              return -0.000605977229f;
            }
          }
        } else {
          if (f[3] <= 3015.5f) {
            return -0.00240957623f;
          } else {
            return 0.000116232878f;
          }
        }
      } else {
        return 0.000399105735f;
      }
    }
    case 62: {
      if (f[0] <= 27.0f) {
        if (f[2] <= 79786.5f) {
          if (f[5] <= 6.5f) {
            if (f[2] <= 3678.5f) {
              return -0.00137385746f;
            } else {
              if (f[2] <= 28579.0f) {
                if (f[3] <= 29509.5f) {
                  if (f[2] <= 24849.5f) {
                    if (f[4] <= 2.18780291f) {
                      return -0.000225110175f;
                    } else {
                      return 0.00322704282f;
                    }
                  } else {
                    return -0.00534052627f;
                  }
                } else {
                  if (f[0] <= 7.0f) {
                    return 9.39040067e-07f;
                  } else {
                    return 0.000810311494f;
                  }
                }
              } else {
                if (f[4] <= 2.4659369f) {
                  if (f[3] <= 38574.5f) {
                    if (f[2] <= 29355.5f) {
                      if (f[4] <= 1.21240151f) {
                        return -0.00469260847f;
                      } else {
                        return 0.000301136215f;
                      }
                    } else {
                      return 0.00142739944f;
                    }
                  } else {
                    return -0.000201714274f;
                  }
                } else {
                  return -0.00627798683f;
                }
              }
            }
          } else {
            return 0.0005007693f;
          }
        } else {
          if (f[0] <= 3.0f) {
            return -0.000116600803f;
          } else {
            return -0.00328400846f;
          }
        }
      } else {
        return 0.000367177278f;
      }
    }
    case 63: {
      if (f[4] <= 1.22994548f) {
        if (f[2] <= 65602.0f) {
          if (f[2] <= 64977.5f) {
            if (f[3] <= 31328.5f) {
              return -0.000750346442f;
            } else {
              if (f[3] <= 46423.0f) {
                return 0.000539278849f;
              } else {
                if (f[2] <= 64275.5f) {
                  return -0.000111891037f;
                } else {
                  if (f[3] <= 70164.5f) {
                    return 0.0131928238f;
                  } else {
                    return -0.00306752054f;
                  }
                }
              }
            }
          } else {
            if (f[3] <= 76106.5f) {
              return 0.005615943f;
            } else {
              return 0.000819432748f;
            }
          }
        } else {
          if (f[3] <= 73758.5f) {
            return -0.00510693794f;
          } else {
            return -0.000654338096f;
          }
        }
      } else {
        if (f[2] <= 57799.0f) {
          if (f[2] <= 30473.5f) {
            if (f[2] <= 7262.0f) {
              if (f[4] <= 2.4659369f) {
                return -0.000772639909f;
              } else {
                return 0.00381921359f;
              }
            } else {
              return 0.000533000328f;
            }
          } else {
            return -9.27376566e-05f;
          }
        } else {
          if (f[2] <= 58406.5f) {
            return 0.00577286882f;
          } else {
            return 0.000621506754f;
          }
        }
      }
    }
    case 64: {
      if (f[0] <= 27.0f) {
        if (f[2] <= 1.00000002e-35f) {
          return 0.0103187532f;
        } else {
          if (f[4] <= 1.77812749f) {
            if (f[2] <= 3678.5f) {
              if (f[4] <= 1.74921805f) {
                return -0.00182337635f;
              } else {
                return 0.00740239574f;
              }
            } else {
              if (f[2] <= 79786.5f) {
                if (f[0] <= 3.0f) {
                  return -0.000521974975f;
                } else {
                  return -2.58170927e-05f;
                }
              } else {
                if (f[0] <= 7.0f) {
                  return -0.000685046113f;
                } else {
                  return -0.00439732652f;
                }
              }
            }
          } else {
            if (f[3] <= 99469.0f) {
              if (f[4] <= 2.4659369f) {
                return 0.000659997638f;
              } else {
                return 0.00310518425f;
              }
            } else {
              if (f[4] <= 2.18780291f) {
                return 0.000850602308f;
              } else {
                return -0.00740477287f;
              }
            }
          }
        }
      } else {
        if (f[0] <= 33.0f) {
          return 0.000413191336f;
        } else {
          if (f[2] <= 70062.5f) {
            if (f[2] <= 69231.0f) {
              return -0.000950565563f;
            } else {
              return -0.0138874292f;
            }
          } else {
            return 0.00470179704f;
          }
        }
      }
    }
    case 65: {
      if (f[5] <= 1.00000002e-35f) {
        if (f[0] <= 27.0f) {
          if (f[2] <= 28579.0f) {
            if (f[3] <= 15987.0f) {
              return -0.000891439987f;
            } else {
              return 0.000222775342f;
            }
          } else {
            if (f[4] <= 2.4659369f) {
              return -0.000255100983f;
            } else {
              if (f[2] <= 42666.5f) {
                return -0.00130725963f;
              } else {
                return -0.0115721389f;
              }
            }
          }
        } else {
          return 0.000812585001f;
        }
      } else {
        if (f[3] <= 88860.5f) {
          if (f[2] <= 76519.0f) {
            if (f[4] <= 1.77812749f) {
              if (f[3] <= 23198.5f) {
                if (f[2] <= 1754.5f) {
                  return -0.00371639365f;
                } else {
                  return -0.000235976771f;
                }
              } else {
                if (f[0] <= 33.0f) {
                  return 0.000209163036f;
                } else {
                  return -0.00101605785f;
                }
              }
            } else {
              return 0.00131258523f;
            }
          } else {
            return -0.00595571216f;
          }
        } else {
          if (f[4] <= 1.1327765f) {
            return -0.00621877424f;
          } else {
            if (f[3] <= 92601.5f) {
              return 0.00285035389f;
            } else {
              return 0.0110624983f;
            }
          }
        }
      }
    }
    case 66: {
      if (f[4] <= 1.22994548f) {
        if (f[2] <= 65602.0f) {
          return -7.17188601e-05f;
        } else {
          if (f[3] <= 80118.5f) {
            if (f[2] <= 71090.5f) {
              if (f[2] <= 69231.0f) {
                if (f[2] <= 66193.5f) {
                  if (f[4] <= 1.15958703f) {
                    if (f[3] <= 73758.5f) {
                      return -0.00711492685f;
                    } else {
                      if (f[4] <= 1.1327765f) {
                        return 0.00717876356f;
                      } else {
                        return -0.00331067791f;
                      }
                    }
                  } else {
                    return -0.0107171104f;
                  }
                } else {
                  return -5.46470483e-05f;
                }
              } else {
                return -0.00765803152f;
              }
            } else {
              if (f[2] <= 72390.5f) {
                return 0.0118357616f;
              } else {
                return -0.00543079596f;
              }
            }
          } else {
            return -0.00030976703f;
          }
        }
      } else {
        if (f[2] <= 57799.0f) {
          if (f[2] <= 30473.5f) {
            if (f[3] <= 24957.0f) {
              return -3.2790906e-05f;
            } else {
              return 0.000635358253f;
            }
          } else {
            return -8.90600035e-05f;
          }
        } else {
          if (f[2] <= 58406.5f) {
            return 0.00531826881f;
          } else {
            return 0.000578534242f;
          }
        }
      }
    }
    case 67: {
      if (f[0] <= 29.0f) {
        if (f[4] <= 1.29975098f) {
          if (f[2] <= 1.00000002e-35f) {
            return 0.00895690841f;
          } else {
            if (f[2] <= 3678.5f) {
              if (f[4] <= 1.20428896f) {
                if (f[3] <= 4228.5f) {
                  return 0.00721309647f;
                } else {
                  return -0.00164401529f;
                }
              } else {
                return -0.0057179797f;
              }
            } else {
              if (f[2] <= 56746.5f) {
                if (f[2] <= 56256.5f) {
                  return -7.59163012e-05f;
                } else {
                  return 0.00183181707f;
                }
              } else {
                if (f[3] <= 63772.5f) {
                  if (f[2] <= 57037.5f) {
                    return -0.00983337712f;
                  } else {
                    if (f[2] <= 58406.5f) {
                      return 0.00240168761f;
                    } else {
                      return -0.0107605873f;
                    }
                  }
                } else {
                  return -0.000366468395f;
                }
              }
            }
          }
        } else {
          return 0.000147134752f;
        }
      } else {
        if (f[0] <= 33.0f) {
          return 0.000526043175f;
        } else {
          if (f[2] <= 70062.5f) {
            if (f[2] <= 69231.0f) {
              return -0.000819031213f;
            } else {
              return -0.0124098767f;
            }
          } else {
            return 0.00431979044f;
          }
        }
      }
    }
    case 68: {
      if (f[0] <= 23.0f) {
        if (f[5] <= 4.5f) {
          if (f[2] <= 79786.5f) {
            if (f[3] <= 18092.5f) {
              return -0.000621056593f;
            } else {
              if (f[2] <= 28579.0f) {
                if (f[4] <= 1.34118497f) {
                  if (f[3] <= 31328.5f) {
                    if (f[2] <= 24849.5f) {
                      if (f[2] <= 14909.0f) {
                        return -0.00335686567f;
                      } else {
                        return -0.000262641381f;
                      }
                    } else {
                      return -0.00259386402f;
                    }
                  } else {
                    return 0.000174739158f;
                  }
                } else {
                  return 0.000612577955f;
                }
              } else {
                if (f[4] <= 2.4659369f) {
                  return -0.000155422168f;
                } else {
                  if (f[2] <= 42666.5f) {
                    return -0.00121740365f;
                  } else {
                    return -0.0106684139f;
                  }
                }
              }
            }
          } else {
            if (f[0] <= 7.0f) {
              return -0.000555735487f;
            } else {
              return -0.00437381096f;
            }
          }
        } else {
          if (f[3] <= 82127.5f) {
            if (f[4] <= 1.07681853f) {
              return 0.00846041784f;
            } else {
              return 0.000418640135f;
            }
          } else {
            return 0.012410876f;
          }
        }
      } else {
        return 0.000195081544f;
      }
    }
    case 69: {
      if (f[0] <= 13.0f) {
        return -0.000149383015f;
      } else {
        if (f[2] <= 85832.0f) {
          if (f[4] <= 2.4659369f) {
            if (f[3] <= 78562.0f) {
              if (f[2] <= 69231.0f) {
                return 8.34199698e-05f;
              } else {
                if (f[2] <= 71090.5f) {
                  return -0.00873392726f;
                } else {
                  return 0.00754928154f;
                }
              }
            } else {
              if (f[4] <= 1.10537148f) {
                return -0.00510984805f;
              } else {
                if (f[4] <= 1.54281402f) {
                  if (f[2] <= 68329.5f) {
                    return 0.0025156623f;
                  } else {
                    if (f[2] <= 70062.5f) {
                      if (f[0] <= 21.0f) {
                        return -0.000565040332f;
                      } else {
                        if (f[4] <= 1.22260153f) {
                          if (f[3] <= 82127.5f) {
                            return -0.00407426698f;
                          } else {
                            return -0.0168871053f;
                          }
                        } else {
                          return 0.00151447154f;
                        }
                      }
                    } else {
                      if (f[4] <= 1.30816251f) {
                        return 0.00272754025f;
                      } else {
                        return -0.00568527977f;
                      }
                    }
                  }
                } else {
                  return -0.00464219244f;
                }
              }
            }
          } else {
            return 0.00322568066f;
          }
        } else {
          return -0.0093684921f;
        }
      }
    }
    case 70: {
      if (f[0] <= 29.0f) {
        if (f[4] <= 1.48282051f) {
          if (f[2] <= 1.00000002e-35f) {
            return 0.00814624593f;
          } else {
            if (f[2] <= 3678.5f) {
              if (f[2] <= 2844.0f) {
                if (f[4] <= 1.43874347f) {
                  return -0.00107751732f;
                } else {
                  return 0.00531869077f;
                }
              } else {
                if (f[3] <= 4228.5f) {
                  return -0.00784319122f;
                } else {
                  return -0.00247565032f;
                }
              }
            } else {
              if (f[5] <= 8.5f) {
                return -9.22826629e-05f;
              } else {
                return 0.000432673614f;
              }
            }
          }
        } else {
          if (f[3] <= 3015.5f) {
            return -0.00288184953f;
          } else {
            if (f[2] <= 72390.5f) {
              return 0.000354769141f;
            } else {
              return -0.00825399684f;
            }
          }
        }
      } else {
        if (f[0] <= 31.0f) {
          return 0.000601203045f;
        } else {
          if (f[2] <= 70062.5f) {
            if (f[2] <= 69231.0f) {
              if (f[3] <= 83425.0f) {
                return -0.000201202592f;
              } else {
                return 0.00709447994f;
              }
            } else {
              return -0.00856053809f;
            }
          } else {
            return 0.00299719096f;
          }
        }
      }
    }
    case 71: {
      if (f[4] <= 1.29975098f) {
        if (f[3] <= 29870.5f) {
          if (f[2] <= 24849.5f) {
            return -0.000401627594f;
          } else {
            if (f[4] <= 1.167014f) {
              if (f[4] <= 1.11502451f) {
                if (f[4] <= 1.10537148f) {
                  return -0.00454711322f;
                } else {
                  return 0.00287597715f;
                }
              } else {
                if (f[2] <= 25958.5f) {
                  return -0.00444110867f;
                } else {
                  return -0.0163215533f;
                }
              }
            } else {
              return 0.00362058661f;
            }
          }
        } else {
          if (f[3] <= 38574.5f) {
            if (f[4] <= 1.08809f) {
              return 0.00495385279f;
            } else {
              if (f[2] <= 33532.5f) {
                if (f[2] <= 29577.0f) {
                  if (f[4] <= 1.2444635f) {
                    if (f[2] <= 28579.0f) {
                      return 2.69560655e-05f;
                    } else {
                      return -0.00231165195f;
                    }
                  } else {
                    return 0.00106653369f;
                  }
                } else {
                  return 0.00130955034f;
                }
              } else {
                if (f[3] <= 37854.0f) {
                  return -0.0152952999f;
                } else {
                  return -0.00014875343f;
                }
              }
            }
          } else {
            return -7.93327684e-05f;
          }
        }
      } else {
        return 0.000164675687f;
      }
    }
    case 72: {
      if (f[0] <= 29.0f) {
        if (f[5] <= 10.5f) {
          if (f[2] <= 1.00000002e-35f) {
            return 0.00851862401f;
          } else {
            if (f[2] <= 1754.5f) {
              return -0.00184253514f;
            } else {
              if (f[4] <= 2.07034707f) {
                if (f[2] <= 7262.0f) {
                  return -0.000635428527f;
                } else {
                  if (f[2] <= 28579.0f) {
                    if (f[3] <= 29509.5f) {
                      if (f[2] <= 24849.5f) {
                        if (f[4] <= 1.2347315f) {
                          return -0.000700357153f;
                        } else {
                          return 0.000134548148f;
                        }
                      } else {
                        return -0.0041157909f;
                      }
                    } else {
                      return 0.000407696891f;
                    }
                  } else {
                    return -0.000110981188f;
                  }
                }
              } else {
                if (f[3] <= 99469.0f) {
                  return 0.00179323275f;
                } else {
                  return -0.00510187185f;
                }
              }
            }
          }
        } else {
          return 0.000632055563f;
        }
      } else {
        if (f[0] <= 33.0f) {
          return 0.00042984876f;
        } else {
          if (f[2] <= 70062.5f) {
            if (f[2] <= 69231.0f) {
              return -0.000765135101f;
            } else {
              return -0.00999932219f;
            }
          } else {
            return 0.00365174816f;
          }
        }
      }
    }
    case 73: {
      if (f[0] <= 13.0f) {
        return -0.000130896914f;
      } else {
        if (f[2] <= 85832.0f) {
          if (f[5] <= 15.5f) {
            if (f[3] <= 78562.0f) {
              if (f[2] <= 69231.0f) {
                if (f[2] <= 68329.5f) {
                  if (f[3] <= 69498.5f) {
                    if (f[2] <= 63417.5f) {
                      return 0.000101200771f;
                    } else {
                      return 0.0102025895f;
                    }
                  } else {
                    if (f[4] <= 1.74921805f) {
                      if (f[2] <= 46517.0f) {
                        return 0.0100293282f;
                      } else {
                        if (f[4] <= 1.51635349f) {
                          return -0.000409198515f;
                        } else {
                          return -0.0102637087f;
                        }
                      }
                    } else {
                      return -0.0160220412f;
                    }
                  }
                } else {
                  return 0.00501302103f;
                }
              } else {
                if (f[2] <= 71090.5f) {
                  return -0.00799524333f;
                } else {
                  return 0.00683202374f;
                }
              }
            } else {
              if (f[4] <= 1.10537148f) {
                return -0.0047500833f;
              } else {
                if (f[4] <= 1.54281402f) {
                  return 0.00128255768f;
                } else {
                  return -0.00406145427f;
                }
              }
            }
          } else {
            return 0.00120740736f;
          }
        } else {
          return -0.00862780161f;
        }
      }
    }
    case 74: {
      if (f[4] <= 1.44255745f) {
        if (f[2] <= 3678.5f) {
          if (f[2] <= 1.00000002e-35f) {
            return 0.00609479571f;
          } else {
            return -0.00177394555f;
          }
        } else {
          return -2.64779697e-05f;
        }
      } else {
        if (f[2] <= 31512.5f) {
          if (f[3] <= 32345.5f) {
            return 0.000122613082f;
          } else {
            if (f[2] <= 12325.0f) {
              return 0.00655959833f;
            } else {
              return 0.000872489431f;
            }
          }
        } else {
          if (f[2] <= 40290.5f) {
            return -0.000850459323f;
          } else {
            if (f[4] <= 2.4659369f) {
              if (f[2] <= 46166.5f) {
                return 0.00142731832f;
              } else {
                if (f[0] <= 11.0f) {
                  if (f[3] <= 82127.5f) {
                    return -0.000691542049f;
                  } else {
                    return 0.00189789191f;
                  }
                } else {
                  if (f[4] <= 1.4923265f) {
                    return 0.000563495907f;
                  } else {
                    if (f[2] <= 47597.5f) {
                      return -0.00898627971f;
                    } else {
                      if (f[2] <= 63417.5f) {
                        return -0.00159146175f;
                      } else {
                        return -0.0099973818f;
                      }
                    }
                  }
                }
              }
            } else {
              return -0.00777733923f;
            }
          }
        }
      }
    }
    case 75: {
      if (f[0] <= 15.0f) {
        return -0.000106124286f;
      } else {
        if (f[2] <= 85832.0f) {
          if (f[3] <= 80118.5f) {
            if (f[2] <= 69231.0f) {
              return 8.42040223e-05f;
            } else {
              if (f[2] <= 71090.5f) {
                return -0.00623936405f;
              } else {
                if (f[2] <= 72390.5f) {
                  return 0.0108952394f;
                } else {
                  return -0.00854905937f;
                }
              }
            }
          } else {
            if (f[4] <= 1.10537148f) {
              return -0.00449679094f;
            } else {
              if (f[4] <= 1.14913052f) {
                if (f[3] <= 82127.5f) {
                  return 0.00115165499f;
                } else {
                  return 0.00802596319f;
                }
              } else {
                if (f[2] <= 68329.5f) {
                  if (f[2] <= 63015.5f) {
                    return -0.000161831471f;
                  } else {
                    return 0.00523502493f;
                  }
                } else {
                  if (f[2] <= 70062.5f) {
                    return -0.00396592163f;
                  } else {
                    if (f[2] <= 71090.5f) {
                      return 0.00475623787f;
                    } else {
                      if (f[4] <= 1.1680755f) {
                        return -0.00507109713f;
                      } else {
                        return 0.00124646158f;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          return -0.00896702805f;
        }
      }
    }
    case 76: {
      if (f[4] <= 1.98430747f) {
        if (f[2] <= 7262.0f) {
          return -0.000565857114f;
        } else {
          if (f[2] <= 31328.5f) {
            if (f[4] <= 1.3224535f) {
              if (f[3] <= 31328.5f) {
                return -0.000386606468f;
              } else {
                if (f[0] <= 5.0f) {
                  return -0.000164225042f;
                } else {
                  if (f[2] <= 27631.0f) {
                    if (f[2] <= 26237.0f) {
                      if (f[2] <= 25680.5f) {
                        if (f[4] <= 1.31860852f) {
                          return 0.000672661804f;
                        } else {
                          return -0.00594037457f;
                        }
                      } else {
                        return -0.00160633373f;
                      }
                    } else {
                      return 0.00147155818f;
                    }
                  } else {
                    return 6.07148447e-05f;
                  }
                }
              }
            } else {
              if (f[0] <= 11.0f) {
                return -0.000825451793f;
              } else {
                return 0.000583191067f;
              }
            }
          } else {
            if (f[3] <= 36773.0f) {
              return 0.00266312673f;
            } else {
              return -6.72507912e-05f;
            }
          }
        }
      } else {
        if (f[3] <= 99469.0f) {
          if (f[3] <= 5226.0f) {
            return -0.00339063069f;
          } else {
            return 0.00145023595f;
          }
        } else {
          return -0.00396013562f;
        }
      }
    }
    case 77: {
      if (f[0] <= 29.0f) {
        if (f[2] <= 79786.5f) {
          if (f[0] <= 3.0f) {
            return -0.000370381968f;
          } else {
            if (f[3] <= 18092.5f) {
              if (f[3] <= 17048.0f) {
                if (f[0] <= 11.0f) {
                  if (f[4] <= 1.39189202f) {
                    return -0.00634519166f;
                  } else {
                    return -0.00115779292f;
                  }
                } else {
                  if (f[4] <= 1.27559596f) {
                    return -0.000845119045f;
                  } else {
                    if (f[4] <= 1.30816251f) {
                      return 0.0019748953f;
                    } else {
                      return -6.11861234e-06f;
                    }
                  }
                }
              } else {
                return -0.0017214204f;
              }
            } else {
              if (f[2] <= 28579.0f) {
                return 0.00023670499f;
              } else {
                return -4.27900251e-05f;
              }
            }
          }
        } else {
          if (f[0] <= 3.0f) {
            return 0.00017986211f;
          } else {
            return -0.00235601594f;
          }
        }
      } else {
        if (f[0] <= 31.0f) {
          return 0.000498767165f;
        } else {
          if (f[5] <= 1.5f) {
            return 0.000830045768f;
          } else {
            if (f[3] <= 86563.5f) {
              return -0.00050550613f;
            } else {
              return 0.00685914772f;
            }
          }
        }
      }
    }
    case 78: {
      if (f[5] <= 13.5f) {
        if (f[2] <= 1754.5f) {
          if (f[2] <= 1.00000002e-35f) {
            return 0.00633812183f;
          } else {
            return -0.00171347455f;
          }
        } else {
          if (f[4] <= 2.07034707f) {
            if (f[2] <= 7262.0f) {
              if (f[2] <= 2844.0f) {
                return 0.00131456964f;
              } else {
                if (f[3] <= 4228.5f) {
                  if (f[4] <= 1.20749003f) {
                    return 0.00149136941f;
                  } else {
                    return -0.00928812695f;
                  }
                } else {
                  if (f[4] <= 1.310179f) {
                    if (f[4] <= 1.25704652f) {
                      return -0.000718705627f;
                    } else {
                      return 0.00307499f;
                    }
                  } else {
                    if (f[4] <= 1.31350553f) {
                      return -0.0096867592f;
                    } else {
                      if (f[2] <= 4183.5f) {
                        if (f[2] <= 3678.5f) {
                          return -0.00203710354f;
                        } else {
                          return 0.00239176868f;
                        }
                      } else {
                        return -0.00158405027f;
                      }
                    }
                  }
                }
              }
            } else {
              return 9.72342507e-08f;
            }
          } else {
            if (f[3] <= 99469.0f) {
              return 0.00153267504f;
            } else {
              return -0.00409642657f;
            }
          }
        }
      } else {
        return 0.00068556044f;
      }
    }
    case 79: {
      if (f[2] <= 65602.0f) {
        if (f[2] <= 64977.5f) {
          if (f[2] <= 64275.5f) {
            return 1.75424854e-05f;
          } else {
            if (f[3] <= 70164.5f) {
              return 0.0118595498f;
            } else {
              if (f[3] <= 74622.5f) {
                if (f[4] <= 1.13554204f) {
                  return -0.013330713f;
                } else {
                  return -0.00454975539f;
                }
              } else {
                return -0.000717814815f;
              }
            }
          }
        } else {
          if (f[3] <= 76106.5f) {
            return 0.0052116229f;
          } else {
            return 0.00107980691f;
          }
        }
      } else {
        if (f[2] <= 66193.5f) {
          if (f[4] <= 1.15958703f) {
            if (f[3] <= 73758.5f) {
              return -0.00651516227f;
            } else {
              return -0.000254283328f;
            }
          } else {
            if (f[0] <= 19.0f) {
              return -0.0034131989f;
            } else {
              return -0.0124264553f;
            }
          }
        } else {
          if (f[4] <= 1.2444635f) {
            return -0.000336926561f;
          } else {
            if (f[5] <= 1.00000002e-35f) {
              if (f[2] <= 67625.5f) {
                return 0.00490548312f;
              } else {
                return -0.00132608019f;
              }
            } else {
              return 0.0093037485f;
            }
          }
        }
      }
    }
    default: return 0.0f;
  }
}

inline float predict_darth_recall(const DarthCommSlot& feature) {
  float f[6];
  f[0] = static_cast<float>(feature.step);
  f[1] = static_cast<float>(feature.found_cnt);
  f[2] = feature.top1_dist;
  f[3] = feature.topk_dist;
  f[4] = feature.gap_ratio;
  f[5] = static_cast<float>(feature.no_improve_count);
  float pred = 0.0f;
  for (int i = 0; i < 80; ++i) {
    pred += predict_darth_tree(i, f);
  }
  if (pred < 0.0f) return 0.0f;
  if (pred > 1.0f) return 1.0f;
  return pred;
}

}  // namespace quiver
