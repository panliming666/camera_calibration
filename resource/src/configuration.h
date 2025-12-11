#ifndef LIBCBDETECT_SETTING_H
#define LIBCBDETECT_SETTING_H

#define NODE_NAME_COEFF      "coeffs"
#define NODE_NAME_INTRINSIC  "intrinsicMat"
#define NODE_NAME_DATE  "date"

#ifndef SVM_PI
#define SVM_PI 3.1415926f
#endif

#define PropertyBuilder(type, name)\
    inline void set_##name( type &v) {\
          name = v;\
    }\
    inline type get_##name() {\
        return name;\
    }\

namespace CalibIntrinsic {

    class Setting {
     public:
        Setting() :
                save_debug_image(true),
                 show_debug_info(true),
                size_square(cv::Size(25, 25)),
                path_image("example_data/test/test.png"),
                path_intrinsic("result/calibration_result.xml"){}

        PropertyBuilder(bool, save_debug_image)
        PropertyBuilder(bool, show_debug_info)
        PropertyBuilder(cv::Size, size_square)
        PropertyBuilder(std::string, path_image)
        PropertyBuilder(std::string, path_intrinsic)

     private:
        bool save_debug_image;
        bool show_debug_info;
        cv::Size size_square;
        std::string path_intrinsic;
        std::string path_image;
    };

}

#endif //LIBCBDETECT_SETTING_H
