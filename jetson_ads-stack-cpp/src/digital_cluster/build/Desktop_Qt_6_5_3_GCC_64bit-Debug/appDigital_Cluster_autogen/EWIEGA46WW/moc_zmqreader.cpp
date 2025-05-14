/****************************************************************************
** Meta object code from reading C++ file 'zmqreader.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.5.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../zmqreader.h"
#include <QtCore/qmetatype.h>

#if __has_include(<QtCore/qtmochelpers.h>)
#include <QtCore/qtmochelpers.h>
#else
QT_BEGIN_MOC_NAMESPACE
#endif


#include <memory>

#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'zmqreader.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.5.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSZMQReaderENDCLASS_t {};
static constexpr auto qt_meta_stringdata_CLASSZMQReaderENDCLASS = QtMocHelpers::stringData(
    "ZMQReader",
    "speedReceived",
    "",
    "speed",
    "batteryReceived",
    "battery",
    "headLightsReceived",
    "headLights",
    "brakeLightReceived",
    "brakeLight",
    "turnLightLeftReceived",
    "turnLightLeft",
    "turnLightRightReceived",
    "turnLightRight",
    "emergencyLightsReceived",
    "emergencyLights",
    "totalDistanceReceived",
    "totalDistance",
    "lkasReceived",
    "lkas",
    "autoPilotReceived",
    "autoPilot",
    "lineLeftReceived",
    "lineLeft",
    "lineRightReceived",
    "lineRight"
);
#else  // !QT_MOC_HAS_STRING_DATA
struct qt_meta_stringdata_CLASSZMQReaderENDCLASS_t {
    uint offsetsAndSizes[52];
    char stringdata0[10];
    char stringdata1[14];
    char stringdata2[1];
    char stringdata3[6];
    char stringdata4[16];
    char stringdata5[8];
    char stringdata6[19];
    char stringdata7[11];
    char stringdata8[19];
    char stringdata9[11];
    char stringdata10[22];
    char stringdata11[14];
    char stringdata12[23];
    char stringdata13[15];
    char stringdata14[24];
    char stringdata15[16];
    char stringdata16[22];
    char stringdata17[14];
    char stringdata18[13];
    char stringdata19[5];
    char stringdata20[18];
    char stringdata21[10];
    char stringdata22[17];
    char stringdata23[9];
    char stringdata24[18];
    char stringdata25[10];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_CLASSZMQReaderENDCLASS_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_CLASSZMQReaderENDCLASS_t qt_meta_stringdata_CLASSZMQReaderENDCLASS = {
    {
        QT_MOC_LITERAL(0, 9),  // "ZMQReader"
        QT_MOC_LITERAL(10, 13),  // "speedReceived"
        QT_MOC_LITERAL(24, 0),  // ""
        QT_MOC_LITERAL(25, 5),  // "speed"
        QT_MOC_LITERAL(31, 15),  // "batteryReceived"
        QT_MOC_LITERAL(47, 7),  // "battery"
        QT_MOC_LITERAL(55, 18),  // "headLightsReceived"
        QT_MOC_LITERAL(74, 10),  // "headLights"
        QT_MOC_LITERAL(85, 18),  // "brakeLightReceived"
        QT_MOC_LITERAL(104, 10),  // "brakeLight"
        QT_MOC_LITERAL(115, 21),  // "turnLightLeftReceived"
        QT_MOC_LITERAL(137, 13),  // "turnLightLeft"
        QT_MOC_LITERAL(151, 22),  // "turnLightRightReceived"
        QT_MOC_LITERAL(174, 14),  // "turnLightRight"
        QT_MOC_LITERAL(189, 23),  // "emergencyLightsReceived"
        QT_MOC_LITERAL(213, 15),  // "emergencyLights"
        QT_MOC_LITERAL(229, 21),  // "totalDistanceReceived"
        QT_MOC_LITERAL(251, 13),  // "totalDistance"
        QT_MOC_LITERAL(265, 12),  // "lkasReceived"
        QT_MOC_LITERAL(278, 4),  // "lkas"
        QT_MOC_LITERAL(283, 17),  // "autoPilotReceived"
        QT_MOC_LITERAL(301, 9),  // "autoPilot"
        QT_MOC_LITERAL(311, 16),  // "lineLeftReceived"
        QT_MOC_LITERAL(328, 8),  // "lineLeft"
        QT_MOC_LITERAL(337, 17),  // "lineRightReceived"
        QT_MOC_LITERAL(355, 9)   // "lineRight"
    },
    "ZMQReader",
    "speedReceived",
    "",
    "speed",
    "batteryReceived",
    "battery",
    "headLightsReceived",
    "headLights",
    "brakeLightReceived",
    "brakeLight",
    "turnLightLeftReceived",
    "turnLightLeft",
    "turnLightRightReceived",
    "turnLightRight",
    "emergencyLightsReceived",
    "emergencyLights",
    "totalDistanceReceived",
    "totalDistance",
    "lkasReceived",
    "lkas",
    "autoPilotReceived",
    "autoPilot",
    "lineLeftReceived",
    "lineLeft",
    "lineRightReceived",
    "lineRight"
};
#undef QT_MOC_LITERAL
#endif // !QT_MOC_HAS_STRING_DATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSZMQReaderENDCLASS[] = {

 // content:
      11,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
      12,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    1,   86,    2, 0x06,    1 /* Public */,
       4,    1,   89,    2, 0x06,    3 /* Public */,
       6,    1,   92,    2, 0x06,    5 /* Public */,
       8,    1,   95,    2, 0x06,    7 /* Public */,
      10,    1,   98,    2, 0x06,    9 /* Public */,
      12,    1,  101,    2, 0x06,   11 /* Public */,
      14,    1,  104,    2, 0x06,   13 /* Public */,
      16,    1,  107,    2, 0x06,   15 /* Public */,
      18,    1,  110,    2, 0x06,   17 /* Public */,
      20,    1,  113,    2, 0x06,   19 /* Public */,
      22,    1,  116,    2, 0x06,   21 /* Public */,
      24,    1,  119,    2, 0x06,   23 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void, QMetaType::QString,    7,
    QMetaType::Void, QMetaType::QString,    9,
    QMetaType::Void, QMetaType::QString,   11,
    QMetaType::Void, QMetaType::QString,   13,
    QMetaType::Void, QMetaType::QString,   15,
    QMetaType::Void, QMetaType::QString,   17,
    QMetaType::Void, QMetaType::QString,   19,
    QMetaType::Void, QMetaType::QString,   21,
    QMetaType::Void, QMetaType::QString,   23,
    QMetaType::Void, QMetaType::QString,   25,

       0        // eod
};

Q_CONSTINIT const QMetaObject ZMQReader::staticMetaObject = { {
    QMetaObject::SuperData::link<QThread::staticMetaObject>(),
    qt_meta_stringdata_CLASSZMQReaderENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSZMQReaderENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSZMQReaderENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<ZMQReader, std::true_type>,
        // method 'speedReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'batteryReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'headLightsReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'brakeLightReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'turnLightLeftReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'turnLightRightReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'emergencyLightsReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'totalDistanceReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'lkasReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'autoPilotReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'lineLeftReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'lineRightReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>
    >,
    nullptr
} };

void ZMQReader::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ZMQReader *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->speedReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 1: _t->batteryReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 2: _t->headLightsReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 3: _t->brakeLightReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 4: _t->turnLightLeftReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 5: _t->turnLightRightReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 6: _t->emergencyLightsReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 7: _t->totalDistanceReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 8: _t->lkasReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 9: _t->autoPilotReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 10: _t->lineLeftReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 11: _t->lineRightReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::speedReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::batteryReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::headLightsReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::brakeLightReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::turnLightLeftReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::turnLightRightReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::emergencyLightsReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::totalDistanceReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::lkasReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 8;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::autoPilotReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 9;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::lineLeftReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 10;
                return;
            }
        }
        {
            using _t = void (ZMQReader::*)(QString );
            if (_t _q_method = &ZMQReader::lineRightReceived; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 11;
                return;
            }
        }
    }
}

const QMetaObject *ZMQReader::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ZMQReader::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSZMQReaderENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return QThread::qt_metacast(_clname);
}

int ZMQReader::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 12)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 12;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 12)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 12;
    }
    return _id;
}

// SIGNAL 0
void ZMQReader::speedReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ZMQReader::batteryReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void ZMQReader::headLightsReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void ZMQReader::brakeLightReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void ZMQReader::turnLightLeftReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void ZMQReader::turnLightRightReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void ZMQReader::emergencyLightsReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void ZMQReader::totalDistanceReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void ZMQReader::lkasReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}

// SIGNAL 9
void ZMQReader::autoPilotReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 9, _a);
}

// SIGNAL 10
void ZMQReader::lineLeftReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 10, _a);
}

// SIGNAL 11
void ZMQReader::lineRightReceived(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 11, _a);
}
QT_WARNING_POP
