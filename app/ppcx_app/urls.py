from django.urls import path

from . import views

urlpatterns = [
    path("images/<int:image_id>/", views.serve_image, name="serve_image"),
    path("dic/upload/", views.upload_dic_h5, name="upload_dic_h5"),
    path("dic/<int:dic_id>/", views.serve_dic_h5, name="serve_dic_h5"),
    path(
        "dic/csv/<int:dic_id>/",
        views.serve_dic_h5_as_csv,
        name="serve_dic_h5_as_csv",
    ),
    path("dic/plot/<int:dic_id>/", views.visualize_dic, name="visualize_dic"),
    path(
        "dic/quiver_image/<int:dic_id>/",
        views.serve_dic_quiver,
        name="serve_dic_quiver",
    ),
    path("dic/visualizer/", views.dic_visualizer, name="dic_visualizer"),
    path("dic/set_label/<int:dic_id>/", views.set_dic_label, name="set_dic_label"),
    path(
        "collapse/visualize/<int:collapse_id>/",
        views.visualize_collapse,
        name="visualize_collapse",
    ),
]
