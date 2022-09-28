---
title: Paper reviews on KSE801 (2022F) 
sidebar: review_sidebar
keywords: reviews
permalink: reviews_kse801_2022.html
toc: true
folder: reviews
summary: Archive to save the reviews from KSE801 (2022F)
---




{% for post in site.posts limit:10 %}
### [{{ post.title }}]({{ post.url | remove: "/"}})
{{ post.date | date: "%b %-d, %Y" }}  
<!-- {% if page.summary %} {{ page.summary | strip_html | strip_newlines | truncate: 160 }} {% else %} {{ post.content | truncatewords: 50 | strip_html }} {% endif %} -->
{% endfor %}

