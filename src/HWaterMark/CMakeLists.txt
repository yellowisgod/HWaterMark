FIND_PACKAGE (Qt5 REQUIRED Core LinguistTools Gui Qml Widgets Quick)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
SET(HWaterMark_MOC_SRCS
      ##ADD MOC FILE HERE ,REALLY IS *.h that CONTAINS Q_OBJECT
)

SET(HWaterMark_SRCS
      ##ADD SRC FILE HERE
      HWaterMarkMain.cpp
)
SET(HWaterMark_UIS
      ##ADD UI FILE HERE
)
SET(HWaterMark_LIBS
      ##ADD LIB FILE HERE
)
SET(HWaterMark_QRCS
      ##ADD QRCS FILE HERE
)
SET(HWaterMark_TRANS
      ##ADD TRANS FILE HERE
)
QT5_WRAP_CPP(HWaterMark_MOC  ${HWaterMark_MOC_SRCS})
QT5_WRAP_UI(HWaterMark_UIC ${HWaterMark_UIS})
QT5_ADD_RESOURCES(HWaterMark_RCC ${HWaterMark_QRCS})
QT5_ADD_TRANSLATION(HWaterMark_QM ${HWaterMark_TRANS})
ADD_DEFINITIONS(-D_REENTRANT -DQT_NO_DEBUG -DQT_SQL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED )
ADD_EXECUTABLE(HWaterMark ${HWaterMark_MOC} ${HWaterMark_SRCS} ${HWaterMark_UIC} ${HWaterMark_RCC} ${HWaterMark_QM}) 
TARGET_LINK_LIBRARIES(HWaterMark ${HWaterMark_LIBS} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Qml Qt5::Quick )