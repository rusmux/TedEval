import importlib
import json
import os
import sqlite3
import sys
import zipfile
from datetime import datetime
from io import BytesIO

from bottle import TEMPLATE_PATH, HTTPResponse, redirect, request, route, run, static_file, template, url
from PIL import Image

from tedeval.web.config import (
    acronym,
    customCSS,
    customJS,
    evaluation_script,
    gt_ext,
    instructions,
    method_params,
    sample_params,
    submit_params,
    title,
)

TEMPLATE_PATH.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "views")),
)


def image_name_to_id(name):
    return name.replace(".jpg", "").replace(".png", "").replace(".gif", "").replace(".bmp", "")


def get_sample_id_from_num(num):
    imagesFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/gt/images.zip"
    archive = zipfile.ZipFile(imagesFilePath, "r")
    current = 0
    for image in archive.namelist():
        if image_name_to_id(image) != False:
            current += 1
            if current == num:
                return image_name_to_id(image)

    return False


def get_sample_from_num(num):
    imagesFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/gt/images.zip"
    archive = zipfile.ZipFile(imagesFilePath, "r")
    current = 0
    for image in archive.namelist():
        if image_name_to_id(image) != False:
            current += 1
            if current == num:
                return image, archive.read(image)

    return False


def get_samples():
    imagesFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/gt/images.zip"
    archive = zipfile.ZipFile(imagesFilePath, "r")
    num_samples = 0
    samples_list = []
    for image in archive.namelist():
        if image_name_to_id(image) != False:
            num_samples += 1
            samples_list.append(image)

    return num_samples, samples_list


@route("/static/:path#.+#", name="static")
def static(path):
    return static_file(
        path,
        root=os.path.abspath(os.path.join(os.path.dirname(__file__), "static")),
    )


@route("/static_custom/:path#.+#", name="static_custom")
def static_custom(path):
    return static_file(
        path,
        root=os.path.abspath(os.path.join(os.path.dirname(__file__), "static_custom")),
    )


@route("/gt/:path#.+#", name="static_gt")
def static_gt(path):
    return static_file(
        path,
        root=os.path.abspath(os.path.join(os.path.dirname(__file__), "gt")),
    )


@route("/favicon.ico")
def favicon():
    return static_file(
        "cvc-ico.png",
        root=os.path.abspath(os.path.join(os.path.dirname(__file__), "static")),
    )


@route("/")
def index():
    _, images_list = get_samples()

    page = int(request.query["p"]) if "p" in request.query else 1
    subm_data = get_all_submissions()

    vars = {
        "url": url,
        "acronym": acronym,
        "title": title,
        "images": images_list,
        "method_params": method_params,
        "page": page,
        "subm_data": subm_data,
        "submit_params": submit_params,
        "instructions": instructions,
        "extension": gt_ext,
    }
    return template("index", vars)


@route("/exit")
def exit() -> None:
    sys.stderr.close()


@route("/method/", methods=["GET"])
def method():
    _, images_list = get_samples()

    results = None
    page = 1
    subm_data = {}

    if "m" in request.query:
        id = request.query["m"]
        submFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/output/results_{id}.zip"

        if os.path.isfile(submFilePath):
            results = zipfile.ZipFile(submFilePath, "r")

        if "p" in request.query:
            page = int(request.query["p"])

        subm_data = get_submission(id)

        if results is None or subm_data is None:
            redirect("/")
    else:
        redirect("/")

    vars = {
        "url": url,
        "acronym": acronym,
        "title": title,
        "images": images_list,
        "method_params": method_params,
        "sample_params": sample_params,
        "results": results,
        "page": page,
        "subm_data": subm_data,
    }
    return template("method", vars)


@route("/sample/")
def sample():
    num_samples, images_list = get_samples()

    sample = int(request.query["sample"])

    methodId = request.query["m"]
    subm_data = get_submission(methodId)

    samplesValues = []

    id = get_sample_id_from_num(sample)
    sampleId = f"{id}.json"

    subms = get_all_submissions()
    for methodId, methodTitle, _, _ in subms:
        zipFolderPath = f"{os.path.dirname(os.path.abspath(__file__))}/output/results_{str(methodId)}"
        sampleFilePath = f"{zipFolderPath}/{sampleId}"

        if os.path.isfile(sampleFilePath) == False:
            submFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/output/results_{str(methodId)}.zip"
            archive = zipfile.ZipFile(submFilePath, "r")

            if os.path.exists(zipFolderPath) == False:
                os.makedirs(zipFolderPath)

            archive.extract(sampleId, zipFolderPath)

        with open(sampleFilePath) as file:
            results = json.loads(file.read())
        # results = json.loads(archive.read(id + ".json"))

        sampleResults = {"id": methodId, "title": methodTitle}
        for k, v in sample_params.items():
            sampleResults[k] = results[k]

        samplesValues.append(sampleResults)

    vars = {
        "url": url,
        "acronym": acronym,
        "title": f"{title} - Sample {sample} : {images_list[sample - 1]}",
        "sample": sample,
        "num_samples": num_samples,
        "subm_data": subm_data,
        "samplesValues": samplesValues,
        "sample_params": sample_params,
        "customJS": customJS,
        "customCSS": customCSS,
    }
    return template("sample", vars)


@route("/sampleInfo/", methods=["GET"])
def get_sample_info():
    methodId = request.query["m"]
    submFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/output/results_{methodId}.zip"
    archive = zipfile.ZipFile(submFilePath, "r")
    id = get_sample_id_from_num(int(request.query["sample"]))
    results = json.loads(archive.read(f"{id}.json"))
    return json.dumps(results)


@route("/image_thumb/", methods=["GET"])
def image_thumb():
    sample = int(request.query["sample"])
    fileName, data = get_sample_from_num(sample)
    ext = fileName.split(".")[-1]

    f = BytesIO(data)
    image = Image.open(f)

    maxsize = (205, 130)
    image.thumbnail(maxsize)
    output = BytesIO()

    if ext == "jpg":
        im_format = "JPEG"
        header = "image/jpeg"
        image.save(output, im_format, quality=80, optimize=True, progressive=True)
    elif ext == "gif":
        im_format = "GIF"
        header = "image/gif"
        image.save(output, im_format)
    elif ext == "png":
        im_format = "PNG"
        header = "image/png"
        image.save(output, im_format, optimize=True)

    contents = output.getvalue()
    output.close()

    body = contents
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"

    return HTTPResponse(body, **headers)


@route("/image/", methods=["GET"])
def image():
    sample = int(request.query["sample"])
    fileName, data = get_sample_from_num(sample)
    ext = fileName.split(".")[-1]
    if ext == "gif":
        header = "image/gif"
    elif ext == "jpg":
        header = "image/jpeg"
    elif ext == "png":
        header = "image/png"

    body = data
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"
    return HTTPResponse(body, **headers)


@route("/gt_image/", methods=["GET"])
def gt_image():
    imagesFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/gt/gt.zip"
    archive = zipfile.ZipFile(imagesFilePath, "r")
    fileName = request.query["sample"]
    ext = fileName.split(".")[-1]
    if ext == "gif":
        header = "image/gif"
    elif ext == "jpg":
        header = "image/jpeg"
    elif ext == "png":
        header = "image/png"

    data = archive.read(fileName)
    body = data
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"
    return HTTPResponse(body, **headers)


@route("/gt_file/", methods=["GET"])
def gt_file():
    imagesFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/gt/gt.zip"
    archive = zipfile.ZipFile(imagesFilePath, "r")
    fileName = request.query["sample"]
    ext = fileName.split(".")[-1]
    if ext == "xml":
        header = "text/xml"

    data = archive.read(fileName)
    body = data
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"
    return HTTPResponse(body, **headers)


@route("/gt_video/", methods=["GET"])
def gt_video():
    imagesFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/gt/images.zip"
    archive = zipfile.ZipFile(imagesFilePath, "r")
    fileName = request.query["sample"]
    ext = fileName.split(".")[-1]
    header = "video/mp4"

    data = archive.read(fileName)
    body = data
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"
    return HTTPResponse(body, **headers)


@route("/subm_image/", methods=["GET"])
def subm_image():
    submFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/output/subm_" + str(request.query["m"]) + ".zip"
    archive = zipfile.ZipFile(submFilePath, "r")
    fileName = request.query["sample"]
    ext = fileName.split(".")[-1]
    if ext == "gif":
        header = "image/gif"
    elif ext == "jpg":
        header = "image/jpeg"
    elif ext == "png":
        header = "image/png"

    data = archive.read(fileName)
    body = data
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"
    return HTTPResponse(body, **headers)


@route("/subm_xml/", methods=["GET"])
def subm_xml():
    submFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/output/subm_" + str(request.query["m"]) + ".zip"
    archive = zipfile.ZipFile(submFilePath, "r")
    fileName = request.query["sample"]
    header = "text/xml"
    data = archive.read(fileName)
    body = data
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"
    return HTTPResponse(body, **headers)


@route("/result_image/", methods=["GET"])
def result_image():
    submFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/output/results_" + str(request.query["m"]) + ".zip"
    archive = zipfile.ZipFile(submFilePath, "r")
    fileName = request.query["name"]
    ext = fileName.split(".")[-1]
    if ext == "gif":
        header = "image/gif"
    elif ext == "jpg":
        header = "image/jpeg"
    elif ext == "png":
        header = "image/png"

    data = archive.read(fileName)
    body = data
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"
    return HTTPResponse(body, **headers)


@route("/result_xml/", methods=["GET"])
def result_xml():
    submFilePath = f"{os.path.dirname(os.path.abspath(__file__))}/output/results_" + str(request.query["m"]) + ".zip"
    archive = zipfile.ZipFile(submFilePath, "r")
    fileName = request.query["name"]
    header = "text/xml"
    data = archive.read(fileName)
    body = data
    headers = {"Content-Type": header}
    if "c" in request.query:
        headers["Cache-Control"] = "public, max-age=3600"
    return HTTPResponse(body, **headers)


@route("/evaluate", method=["POST", "GET"])
def evaluate():
    id = 0
    submFile = request.files.get("submissionFile")

    if submFile is None:
        resDict = {"calculated": False, "Message": "No file selected"}
        if request.query["json"] == "1":
            return json.dumps(resDict)
        vars = {"url": url, "title": f"Method Upload {title}", "resDict": resDict}
    else:
        name, ext = os.path.splitext(submFile.filename)
        if ext not in f".{gt_ext}":
            resDict = {
                "calculated": False,
                "Message": f"File not valid. A {gt_ext.upper()} file is required.",
            }
            if request.query["json"] == "1":
                return json.dumps(resDict)
            vars = {"url": url, "title": f"Method Upload {title}", "resDict": resDict}
            return template("upload", vars)

        p = {
            "g": f"{os.path.dirname(os.path.abspath(__file__))}/gt/gt.{gt_ext}",
            "s": f"{os.path.dirname(os.path.abspath(__file__))}/output/subm.{gt_ext}",
            "o": f"{os.path.dirname(os.path.abspath(__file__))}/output",
        }

        for k, _ in submit_params.items():
            p["p"][k] = request.forms.get(k)

        if os.path.isfile(p["s"]):
            os.remove(p["s"])

        submFile.save(p["s"])

        module = importlib.import_module(evaluation_script)
        resDict = rrc_evaluation_funcs.main_evaluation(
            p,
            module.default_evaluation_params,
            module.validate_data,
            module.evaluate_method,
        )

        if resDict["calculated"] == True:
            dbPath = f"{os.path.dirname(os.path.abspath(__file__))}/output/submits"
            conn = sqlite3.connect(dbPath)
            cursor = conn.cursor()

            submTitle = request.forms.get("title")
            if submTitle == "":
                submTitle = "unnamed"

            cursor.execute(
                "INSERT INTO submission(title,sumbit_date,results) VALUES(?,?,?)",
                (
                    submTitle,
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    json.dumps(resDict["method"]),
                ),
            )
            conn.commit()
            id = cursor.lastrowid

            os.rename(p["s"], p["s"].replace(f"subm.{gt_ext}", f"subm_{str(id)}.{gt_ext}"))
            os.rename(p["o"] + "/results.zip", p["o"] + "/results_" + str(id) + ".zip")

            conn.close()

        if request.query["json"] == "1":
            return json.dumps(
                {
                    "calculated": resDict["calculated"],
                    "Message": resDict["Message"],
                    "id": id,
                },
            )
        vars = {
            "url": url,
            "title": f"Method Upload {title}",
            "resDict": resDict,
            "id": id,
        }
    return template("upload", vars)


@route("/delete_all", method="POST")
def delete_all() -> None:
    output_folder = f"{os.path.dirname(os.path.abspath(__file__))}/output"
    try:
        for root, dirs, files in os.walk(output_folder, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    except Exception:
        print("Unexpected error:", sys.exc_info()[0])


@route("/delete_method", method="POST")
def delete_method() -> None:
    idx = request.forms.get("idx")

    try:
        output_folder = f"{os.path.dirname(os.path.abspath(__file__))}/output/results_{idx}"
        if os.path.isdir(output_folder):
            for root, dirs, files in os.walk(output_folder, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            os.rmdir(output_folder)
        subm_file = f"{os.path.dirname(os.path.abspath(__file__))}/output/results_{idx}.{gt_ext}"
        results_file = f"{os.path.dirname(os.path.abspath(__file__))}/output/subm_{idx}.zip"
        os.remove(subm_file)
        os.remove(results_file)
    except Exception:
        print("Unexpected error:", sys.exc_info()[0])

    dbPath = f"{os.path.dirname(os.path.abspath(__file__))}/output/submits"
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM submission WHERE idx=?", (idx))
    conn.commit()
    conn.close()


@route("/edit_method", method="POST")
def edit_method() -> None:
    id = request.forms.get("id")
    name = request.forms.get("name")

    dbPath = f"{os.path.dirname(os.path.abspath(__file__))}/output/submits"
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()
    cursor.execute("UPDATE submission SET title=? WHERE id=?", (name, id))
    conn.commit()
    conn.close()


def get_all_submissions():
    dbPath = f"{os.path.dirname(os.path.abspath(__file__))}/output/submits"
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS submission(id integer primary key autoincrement, title varchar(50), sumbit_date varchar(12),results TEXT)""",
    )
    conn.commit()

    cursor.execute("SELECT id,title,sumbit_date,results FROM submission")
    sumbData = cursor.fetchall()
    conn.close()
    return sumbData


def get_submission(id):
    dbPath = f"{os.path.dirname(os.path.abspath(__file__))}/output/submits"
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS submission(id integer primary key autoincrement, title varchar(50), sumbit_date varchar(12),results TEXT)""",
    )
    conn.commit()

    cursor.execute(
        "SELECT id,title,sumbit_date,results FROM submission WHERE id=?",
        (id,),
    )
    sumbData = cursor.fetchone()
    conn.close()

    return sumbData


if __name__ == "__main__":
    evalModule = importlib.import_module(evaluation_script)
    try:
        for module, alias in evalModule.evaluation_imports().items():
            importlib.import_module(module)
    except ImportError:
        print(f"Script {evaluation_script}. Required module ({module}) not found.")
        if module == "Polygon":
            print("Install it with: pip3 install Polygon3")
        else:
            print(f"Install it with: pip install {module}")
        sys.exit(101)

    print("***********************************************")
    print("RRC Standalone Task")
    print("-----------------------------------------------")
    print(
        'Command line client:\ncurl -F "submissionFile=submit.zip" http://127.0.0.1:8080/evaluate',
    )
    print("\nGUI client:firefox http://127.0.0.1:8080")
    print("-----------------------------------------------")
    run(host="0.0.0.0", port=8080, debug=True)
