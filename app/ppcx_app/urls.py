from django.urls import path

from . import views

urlpatterns = [
    path("images/<int:image_id>/", views.serve_image, name="serve_image"),
    path("dic/<int:dic_id>/", views.serve_dic_h5, name="serve_dic_h5"),
    path(
        "dic/<int:dic_id>/csv/", views.serve_dic_h5_as_csv, name="serve_dic_h5_as_csv"
    ),
    path("dic/<int:dic_id>/plot/", views.visualize_dic, name="visualize_dic"),
    path(
        "dic/<int:dic_id>/quiver_image/",
        views.serve_dic_quiver,
        name="serve_dic_quiver",
    ),
    path("dic/visualizer/", views.dic_visualizer, name="dic_visualizer"),
    path("dic/<int:dic_id>/set_label/", views.set_dic_label, name="set_dic_label"),
    path(
        "API/collapse/<int:collapse_id>/visualize/",
        views.visualize_collapse,
        name="visualize_collapse",
    ),
]
